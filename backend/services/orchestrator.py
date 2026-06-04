"""
Orchestrator service - coordinates ML, GenAI, and storage services
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def run_analysis(values: List[float]) -> Dict[str, Any]:
    """
    Complete analysis pipeline:
    1. ML prediction
    2. LLM explanation
    3. Save to history
    
    Args:
        values: [usage_hours, support_tickets, tenure_months]
    
    Returns:
        Complete analysis result
    """
    try:
        # ====== STEP 1: ML PREDICTION ======
        from backend.services.ds_service import predict
        
        logger.info("🔄 Step 1: Running ML prediction...")
        ds_output = predict(values)
        logger.info(f"✅ ML prediction complete: {ds_output['risk_level']} risk")
        
        # ====== STEP 2: LLM EXPLANATION ======
        from backend.services.genai_service import generate_explanation
        
        logger.info("🔄 Step 2: Generating explanation...")
        explanation = generate_explanation(ds_output)
        logger.info("✅ Explanation generated (LLM or fallback)")
        
        # ====== STEP 3: SAVE HISTORY ======
        from backend.services.history_service import save_record
        
        logger.info("🔄 Step 3: Saving to history...")
        save_record(values, ds_output)
        logger.info("✅ Record saved")
        
        # ====== RETURN COMBINED RESULT ======
        result = {
            "ds_output": ds_output,
            "explanation": explanation
        }
        
        logger.info("✅ Analysis pipeline complete")
        return result
    
    except Exception as e:
        logger.error(f"❌ Analysis pipeline failed: {e}", exc_info=True)
        raise
