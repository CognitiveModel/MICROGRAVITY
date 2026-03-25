import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from coding_agent.ui_agent.planning.experiential_memory import ExperientialMemory
from coding_agent.ui_agent.planning.task_schema import NodeType

class TestExperientialLearning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "test_memory_dir"
        self.mem = ExperientialMemory(self.temp_dir)

    def tearDown(self):
        if os.path.exists(os.path.join(self.temp_dir, "memory.json")):
            os.remove(os.path.join(self.temp_dir, "memory.json"))
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_variable_and_constant_deduction(self):
        # Episode 1: Send email to alice
        ep1_steps = [
            {"action": "click", "target": "compose_button"},
            {"action": "type", "target": "to_field", "method": "alice@example.com"},
            {"action": "click", "target": "send_button"}
        ]
        
        # Episode 2: Send email to bob
        ep2_steps = [
            {"action": "click", "target": "compose_button"},
            {"action": "type", "target": "to_field", "method": "bob@example.com"},
            {"action": "click", "target": "send_button"}
        ]
        
        # Record episodes
        self.mem.record_episode("Send simple email", "MailClient", "MAIL", ep1_steps, success=True)
        self.mem.record_episode("Send simple email", "MailClient", "MAIL", ep2_steps, success=True)
        
        # Verify schema
        task_id = "schema_send_simple_email"
        self.assertIn(task_id, self.mem.task_schemas)
        schema = self.mem.task_schemas[task_id]
        
        # We expect 2 constants (click compose, click send) with empty text, actually wait, missing methods are constants?
        # In our implementation: `elif method:` it creates a constant. But if method is empty, no constant.
        # But we can verify 1 variable is created!
        
        vars_created = list(schema.variables.values())
        self.assertEqual(len(vars_created), 1)
        
        var = vars_created[0]
        self.assertIn("alice@example.com", var.examples)
        self.assertIn("bob@example.com", var.examples)
        
        # Execution graph has 3 nodes
        self.assertEqual(len(schema.execution_graph), 3)
        self.assertEqual(schema.execution_graph[0].action_type, "click")
        self.assertEqual(schema.execution_graph[1].action_type, "type")
        self.assertEqual(schema.execution_graph[2].action_type, "click")
        
        # The second node should reference the variable
        # Execute third episode with radically different step count to trigger self-realization divergence
        ep3_steps = [
            {"action": "click", "target": "compose_button"},
            {"action": "click", "target": "add_attachment"},
            {"action": "type", "target": "file_path", "method": "C:/doc.pdf"},
            {"action": "click", "target": "confirm_attachment"},
            {"action": "type", "target": "to_field", "method": "charlie@example.com"},
            {"action": "click", "target": "send_button"}
        ]
        self.mem.record_episode("Send simple email", "MailClient", "MAIL", ep3_steps, success=True)
        
        # We should now have a variant schema spawned
        schemas = self.mem.task_schemas
        # It's named dynamically with timestamp, so we search by parent_schema_id
        variants = [s for s in schemas.values() if s.parent_schema_id == task_id]
        self.assertEqual(len(variants), 1)
        self.assertFalse(variants[0].differentiation.is_human_guided)
        self.assertIn("diverged significantly", variants[0].differentiation.justification)
        
        # Test Human-in-the-loop differentiation
        human_variant_id = self.mem.differentiate_schema_human_loop(
            base_task_id=task_id, 
            new_task_name="Send urgent email",
            justification="Urgent emails require cc-ing manager",
            trigger_condition="User explicitly requested separation"
        )
        self.assertIn(human_variant_id, self.mem.task_schemas)
        human_variant = self.mem.task_schemas[human_variant_id]
        self.assertTrue(human_variant.differentiation.is_human_guided)
        self.assertEqual(human_variant.differentiation.trigger_condition, "User explicitly requested separation")

    def test_proactive_planning(self):
        """Test proposing and discarding a planned schema proactively."""
        # Create a manually planned schema
        from coding_agent.ui_agent.planning.task_schema import ParameterizedTaskSchema, ActionNode

        schema = ParameterizedTaskSchema(
            task_id="schema_draft_email",
            task_name="Draft email",
            app_class="MAIL",
            execution_graph=[ActionNode("n1", "click", "compose_button")]
        )
        # Propose it
        self.mem.propose_planned_schema(schema)
        
        self.assertIn("schema_draft_email", self.mem.task_schemas)
        self.assertTrue(self.mem.task_schemas["schema_draft_email"].is_planned_only)
        
        # Discard it
        discarded = self.mem.discard_planned_schema("schema_draft_email")
        self.assertTrue(discarded)
        self.assertNotIn("schema_draft_email", self.mem.task_schemas)

    def test_error_and_path_tracking(self):
        """Test that failed episodes with eventual success generate error records and paths."""
        task_id = "schema_test_tracked_task"
        
        # We need to monkeypatch the PathAnalyzerAgent so we don't hit the real Gemini API
        from coding_agent.subagents.path_analyzer_agent import PathAnalyzerAgent
        from coding_agent.ui_agent.planning.task_schema import ErrorRecord
        
        original_init = PathAnalyzerAgent.__init__
        original_analyze = PathAnalyzerAgent.analyze_episode
        
        def mock_init(self_instance, model_name=None):
            self_instance.model_name = "mock-model"
            self_instance.client = None
            
        def mock_analyze(self_instance, episode_steps):
            return ErrorRecord(
                error_id="err_mock123",
                description="Mock error finding bad button",
                recovery_summary="Found the right button",
                failed_steps=["click on 'wrong button'"]
            )
            
        PathAnalyzerAgent.__init__ = mock_init
        PathAnalyzerAgent.analyze_episode = mock_analyze
        
        try:
            ep_steps = [
                {"action": "click", "target": "wrong button", "success": False},
                {"action": "click", "target": "right button", "success": True},
                {"action": "click", "target": "submit", "success": True}
            ]
            self.mem.record_episode("Test tracked task", "TESTAPP", "TEST", ep_steps, success=True)
            
            schema = self.mem.task_schemas[task_id]
            self.assertIn("err_mock123", schema.known_errors)
            self.assertEqual(schema.known_errors["err_mock123"].description, "Mock error finding bad button")
            
            # Should have registered at least one valid path
            self.assertGreater(len(schema.valid_paths), 0)
            
            # The execution graph should just be the successful steps
            self.assertEqual(len(schema.execution_graph), 2)
            self.assertEqual(schema.execution_graph[0].target_identifier, "right button")
        finally:
            PathAnalyzerAgent.__init__ = original_init
            PathAnalyzerAgent.analyze_episode = original_analyze

if __name__ == "__main__":
    unittest.main()
