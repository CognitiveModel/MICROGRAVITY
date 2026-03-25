"""
MICROGRAVITY WISDOM ENGINE — 7R Pipeline

This module implements the formal 7R Cognitive Pipeline (R1-R7) as defined 
in the CSUSE_CORE architecture.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Wisdom Enums
# ═══════════════════════════════════════════════════════════════

class SophisticationLevel(str, Enum):
    REFLEX = "REFLEX"
    STRUCTURAL = "STRUCTURAL"
    GENERATIVE = "GENERATIVE"

class WisdomInterruptType(str, Enum):
    OVERSIMPLIFICATION = "oversimplification"
    SEMANTIC_DRIFT = "semantic_drift"
    HARD_FAILURE = "hard_failure"
    DISCERNMENT_DEFICIT = "discernment_deficit"

class HopType(str, Enum):
    R1_CHARACTERIZATION = "r1_characterization"
    R2_CLASSIFICATION = "r2_classification"
    R3_TERMINOLOGY = "r3_terminology"
    R4_BOUNDING = "r4_bounding"
    R5_CONSTRUCTION = "r5_construction"
    R6_CONVICTION = "r6_conviction"

# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

class WisdomInterrupt(BaseModel):
    interrupt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interrupt_type: WisdomInterruptType
    reason: str
    target_hop: HopType

class HopResult(BaseModel):
    hop_type: HopType
    content: str
    discernment_points: list[str] = Field(default_factory=list)
    hardness_grade: int = 0
    interrupts_raised: list[WisdomInterrupt] = Field(default_factory=list)

# ═══════════════════════════════════════════════════════════════
#  The 7R Wisdom Engine
# ═══════════════════════════════════════════════════════════════

class WisdomEngine:
    """Executes the 7R Wisdom Pipeline."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.active_interrupts: list[WisdomInterrupt] = []

    async def execute_pipeline(self, task_description: str, min_sophistication: SophisticationLevel) -> dict[str, Any]:
        """Runs R1 through R6 (R7 is Activation via the Agent's execution loop)."""
        logger.info(f"Starting 7R Wisdom Pipeline for task. Target Sophistication: {min_sophistication}")
        self.active_interrupts.clear()

        # R1: Characterization (Perception)
        r1 = await self._run_r1(task_description)
        self._check_interrupts(r1)

        # R2-R3: Classification & Terminology
        r23 = await self._run_r2_r3(r1.content)
        self._check_interrupts(r23)

        # R4: Bounding
        r4 = await self._run_r4(r23.content)
        self._check_interrupts(r4)

        # R5: Construction
        r5 = await self._run_r5(task_description, r4.content)
        self._check_interrupts(r5)

        # R6: Conviction (Self-Adversarial grading)
        r6 = await self._run_r6(r5.content)
        self._check_interrupts(r6)

        # Final checks
        if r6.hardness_grade < 7:
            self._raise_interrupt(
                WisdomInterruptType.SEMANTIC_DRIFT,
                "Conviction score fell below Hardness 7. Strategy is too tentative.",
                HopType.R6_CONVICTION
            )

        return {
            "task": task_description,
            "r1_entity_map": r1.content,
            "r3_named_intent": r23.content,
            "r4_constraints": r4.content,
            "r5_operator_plan": r5.content,
            "r6_conviction_report": r6.content,
            "hardness_grade": r6.hardness_grade,
            "interrupts_handled": len(self.active_interrupts)
        }

    # --- Phase Logic ---

    async def _run_r1(self, task: str) -> HopResult:
        prompt = f"R1 CHARACTERIZATION: Normalize the raw signals in this task into an EntityMap.\nTASK: {task}"
        resp = await self._query_llm(prompt)
        
        interrupts = []
        if len(resp.split()) < 5:
            interrupts.append(WisdomInterrupt(
                interrupt_type=WisdomInterruptType.OVERSIMPLIFICATION,
                reason="R1 failed to distinguish signal from background noise.",
                target_hop=HopType.R1_CHARACTERIZATION
            ))

        return HopResult(hop_type=HopType.R1_CHARACTERIZATION, content=resp, interrupts_raised=interrupts)

    async def _run_r2_r3(self, entity_map: str) -> HopResult:
        prompt = f"R2/R3 CLASSIFICATION & TERMINOLOGY: Classify entities and align to Domain Language.\nMAP: {entity_map}"
        resp = await self._query_llm(prompt)
        return HopResult(hop_type=HopType.R3_TERMINOLOGY, content=resp)

    async def _run_r4(self, named_intent: str) -> HopResult:
        prompt = f"R4 BOUNDING: Define Spatial, Logical, and Temporal boundaries for this intent.\nINTENT: {named_intent}"
        resp = await self._query_llm(prompt)
        return HopResult(hop_type=HopType.R4_BOUNDING, content=resp)

    async def _run_r5(self, task: str, boundaries: str) -> HopResult:
        prompt = f"R5 CONSTRUCTION: Synthesize the execution plan and project an anticipated result.\nTASK: {task}\nBOUNDARIES: {boundaries}"
        resp = await self._query_llm(prompt)
        return HopResult(hop_type=HopType.R5_CONSTRUCTION, content=resp)

    async def _run_r6(self, operator_plan: str) -> HopResult:
        prompt = (
            f"R6 CONVICTION: Perform a self-adversarial falsification on this plan.\n"
            f"Assign a HardnessGrade from 1-10 based on likelihood of survival.\n"
            f"PLAN: {operator_plan}"
        )
        resp = await self._query_llm(prompt)
        
        grade = 10
        if "grade" in resp.lower() or "hardnessgrade" in resp.lower():
            # Very naive extraction of grade for demo
            try:
                for word in resp.split():
                    if word.isdigit():
                        grade = int(word)
                        break
            except Exception:
                pass

        return HopResult(hop_type=HopType.R6_CONVICTION, content=resp, hardness_grade=grade)

    # --- Monitors ---

    def _check_interrupts(self, result: HopResult):
        for interrupt in result.interrupts_raised:
            logger.warning(f"[WISDOM INTERRUPT] {interrupt.interrupt_type.name}: {interrupt.reason}")
            self.active_interrupts.append(interrupt)

    def _raise_interrupt(self, type: WisdomInterruptType, reason: str, hop: HopType):
        interrupt = WisdomInterrupt(interrupt_type=type, reason=reason, target_hop=hop)
        logger.warning(f"[WISDOM INTERRUPT] {type.name}: {reason}")
        self.active_interrupts.append(interrupt)

    async def _query_llm(self, prompt: str) -> str:
        """Bridge to the local LLM Client."""
        if self.llm_client:
            return await self.llm_client.generate_content(prompt)
        return f"Mock Validated Response for: {prompt[:30]}..."
