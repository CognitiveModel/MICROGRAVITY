"""Wisdom Interrupt Monitor."""

import uuid
from datetime import datetime

from loguru import logger # type: ignore

from .blob_store import BlobStore # type: ignore
from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .models.wisdom import WisdomInterrupt, WisdomInterruptType # type: ignore


class InterruptMonitor:
    """
    Monitors operations for logical fallacies, shallow reasoning, or missing boundaries
    and throws Wisdom Interrupts to force a cognitive rewind.
    Implements Swarm Architecture §6.
    """

    def __init__(self, lmdb: LMDBStore, graph: GraphStore, blobs: BlobStore):
        self.lmdb = lmdb
        self.graph = graph
        self.blobs = blobs
        
        # Simple heuristic triggers for MVP - normally these would be trained 
        # or managed dynamically via the TriggerEngine.
        self._heuristics = {
            "obviously": WisdomInterruptType.PREMATURE_CONVICTION,
            "always": WisdomInterruptType.FALSE_DICHOTOMY,
            "never": WisdomInterruptType.FALSE_DICHOTOMY,
            "everyone knows": WisdomInterruptType.EPISTEMIC_TRESPASSING,
            "just": WisdomInterruptType.SHALLOWNESS,
            "simply put": WisdomInterruptType.SHALLOWNESS,
            "because it is": WisdomInterruptType.CIRCULAR_REASONING,
        }

    def scan_chunk(self, chunk: str, task_id: str) -> WisdomInterrupt | None:
        """
        Scan a text chunk (e.g., from an LLM stream or tool output) 
        for patterns that warrant an interrupt.
        """
        # 1. Very basic heuristic scan
        lower_chunk = chunk.lower()
        for phrase, int_type in self._heuristics.items():
            if f" {phrase} " in f" {lower_chunk} ":
                return self.fire_interrupt(int_type, task_id, context=f"Detected trigger phrase: '{phrase}'")

        # 2. LMDB fast-state checks
        # e.g., if we've cycled through 3 tool errors in a row without changing approach
        err_count = self.lmdb.get(f"sprocess:error_chain:{task_id}", 0)
        if err_count >= 3:
            return self.fire_interrupt(
                WisdomInterruptType.CONFIRMATION_BIAS, 
                task_id, 
                context="Agent is repeating failing actions without re-evaluating."
            )

        return None

    def fire_interrupt(self, interrupt_type: WisdomInterruptType, task_id: str, context: str = "") -> WisdomInterrupt:
        """
        Fire an interrupt, initiating the 7-step resolution protocol.
        """
        int_id = f"int_{uuid.uuid4().hex[:8]}" # type: ignore
        interrupt = WisdomInterrupt(
            interrupt_id=int_id,
            type=interrupt_type,
            task_id=task_id,
            fired_at=datetime.utcnow(),
            triggering_content=context
        )
        
        logger.warning(
            "⚡ WISDOM INTERRUPT [{}]: Task {} halted. Reason: {}", 
            interrupt.type.name, task_id, context # type: ignore
        )

        # 1. State Store Logging
        self.lmdb.put(f"interrupt:active:{task_id}", {
            "id": int_id,
            "type": interrupt.type.value,
            "fired_at": interrupt.fired_at.isoformat(),
            "status": "ACTIVE"
        })
        
        # 2. Detailed Trace to Blob Storage (for post-mortem/learning)
        trace_data = {
            "interrupt_id": int_id,
            "task_id": task_id,
            "type": interrupt.type.value,
            "context": context,
            "timestamp": interrupt.fired_at.isoformat(),
            "pipeline_state": self.lmdb.get(f"sprocess:pipeline_phase:{task_id}")
        }
        self.blobs.log_interrupt(int_id, trace_data)
        
        # 3. Apply Resolution Protocol (e.g. Rewind)
        interrupt.hops_rewound = self._apply_resolution(interrupt)
        
        return interrupt

    def _apply_resolution(self, interrupt: WisdomInterrupt) -> int:
        """
        Apply the resolution strategy mapped to the interrupt type.
        Returns the number of logical steps rewound.
        """
        rewind_steps = 0
        
        # E.g., if premature conviction, we step back to CONSTRUCTION phase
        if interrupt.type in [WisdomInterruptType.PREMATURE_CONVICTION, WisdomInterruptType.SHALLOWNESS]:
            state = self.lmdb.get(f"sprocess:pipeline_phase:{interrupt.task_id}")
            if state and state.get("phase") in ["CONVICTION", "ACTIVATION"]:
                state["phase"] = "CONSTRUCTION" # Force deepening
                self.lmdb.put(f"sprocess:pipeline_phase:{interrupt.task_id}", state)
                rewind_steps = 1
                
        # Other interrupt types would trigger context re-loading, etc.
        
        interrupt.resolution_status = "RESOLVED"
        # Clear active flag
        self.lmdb.delete(f"interrupt:active:{interrupt.task_id}")
        
        return rewind_steps

    def record_error(self, task_id: str) -> None:
        """Track procedural errors to catch confirmation bias loops."""
        key = f"sprocess:error_chain:{task_id}"
        count = self.lmdb.get(key, 0)
        self.lmdb.put(key, count + 1) # type: ignore
        
    def clear_errors(self, task_id: str) -> None:
        """Reset procedural error chain on success."""
        self.lmdb.delete(f"sprocess:error_chain:{task_id}")
