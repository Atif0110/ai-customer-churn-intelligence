import logging
import time
from typing import Dict, Any, Optional
from backend.llm.factory import get_llm
from backend.core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class EnhancedGenAIService:
    """
    Generate business-friendly explanations with fallback strategies
    Implements retry logic and graceful degradation
    """
    
    def __init__(self, max_retries: int = 2, timeout: int = 30):
        """
        Initialize GenAI service
        
        Args:
            max_retries: Number of retry attempts on failure
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        
        try:
            self.llm = get_llm()
            logger.info("✅ LLM service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ LLM initialization warning: {e}")
            self.llm = None
    
    def generate_explanation(self, ds_output: Dict[str, Any]) -> str:
        """
        Generate AI explanation of churn risk with fallback
        
        Args:
            ds_output: ML model output containing:
                - churn_probability: float (0-1)
                - risk_level: str ("Low", "Medium", "High")
                - drivers: List[str] of risk factors
        
        Returns:
            Human-readable explanation (always returns something)
        """
        try:
            # Try LLM with retries
            return self._generate_with_retry(ds_output)
        
        except LLMServiceError as e:
            logger.warning(f"⚠️ LLM service failed: {e}. Using fallback.")
            return self._fallback_explanation(ds_output)
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in explanation generation: {e}")
            return self._fallback_explanation(ds_output)
    
    def _generate_with_retry(self, ds_output: Dict[str, Any]) -> str:
        """
        Try LLM multiple times with exponential backoff
        
        Raises:
            LLMServiceError: If all retries exhausted
        """
        last_error: Optional[str] = None
        
        for attempt in range(self.max_retries):
            try:
                prompt = self._build_optimized_prompt(ds_output)
                
                logger.info(f"🔄 LLM attempt {attempt + 1}/{self.max_retries}...")
                
                start_time = time.time()
                result = self.llm.generate(prompt)
                elapsed = time.time() - start_time
                
                # Validate response
                if result and len(result.strip()) > 20:
                    logger.info(
                        f"✅ LLM succeeded on attempt {attempt + 1}/{self.max_retries} "
                        f"in {elapsed*1000:.1f}ms"
                    )
                    return result
                else:
                    last_error = f"Response too short: {len(result) if result else 0} chars"
                    logger.warning(f"⚠️ Attempt {attempt + 1}: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"⚠️ LLM attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, etc.
                    logger.info(f"⏳ Retrying in {wait_time} second(s)...")
                    time.sleep(wait_time)
        
        # All retries exhausted
        error_msg = f"All {self.max_retries} retry attempts failed: {last_error}"
        logger.error(f"❌ {error_msg}")
        raise LLMServiceError(error_msg)
    
    def _build_optimized_prompt(self, ds_output: Dict[str, Any]) -> str:
        """
        Build high-quality prompt for better LLM output
        Optimized for conciseness and clarity
        """
        drivers = ", ".join(ds_output.get("drivers", []))
        prob = ds_output.get("churn_probability", 0)
        risk = ds_output.get("risk_level", "Unknown")
        
        prompt = f"""You are an expert customer success strategist analyzing churn risk.

CUSTOMER STATUS:
- Churn Risk Score: {prob*100:.1f}%
- Risk Level: {risk}
- Key Risk Drivers: {drivers}

YOUR TASK - Provide 3 things:

1. **Why They Might Leave** (2 sentences max)
   - Explain churn drivers from customer perspective
   - Use business language, not metrics

2. **Two Retention Actions** (2 specific bullets)
   - What should success team do THIS WEEK?
   - Make them concrete and measurable
   - Prioritize based on risk level

3. **Business Impact** (1 sentence)
   - What does losing this customer cost?
   - Frame as revenue or strategic impact

TONE GUIDELINES:
- Conversational but professional
- Data-driven and specific
- Action-focused and urgent
- Avoid jargon; use clear business language

RISK-BASED URGENCY:
- High Risk: Emphasize immediate action needed
- Medium Risk: Recommend proactive engagement
- Low Risk: Focus on expansion opportunity"""
        
        return prompt
    
    def _fallback_explanation(self, ds_output: Dict[str, Any]) -> str:
        """
        Rule-based fallback explanation when LLM fails
        Uses templates + customer data to generate meaningful response
        Ensures users always get helpful output
        """
        prob = ds_output.get("churn_probability", 0)
        risk = ds_output.get("risk_level", "Unknown")
        drivers = ds_output.get("drivers", [])
        
        # Define detailed explanations for common drivers
        driver_explanations = {
            "Low product engagement": (
                "The customer isn't using the product enough to realize its value. "
                "They may not understand key features or haven't fully onboarded yet. "
                "Typical engagement increase: 25-35% reduction in churn."
            ),
            "High support load": (
                "Frequent support requests indicate the customer is struggling. "
                "This signals product friction, unclear documentation, or unmet needs. "
                "Resolving core issues typically reduces churn by 30-40%."
            ),
            "Early-stage customer": (
                "Customers in their first 18 months have 2-3x higher churn. "
                "This is the critical retention window. Strong onboarding is key. "
                "Success programs reduce early-stage churn by 40-50%."
            ),
            "Stable customer profile": (
                "This customer shows healthy engagement and low risk factors. "
                "Focus on deepening the relationship and expanding usage. "
                "Opportunity to increase lifetime value."
            ),
        }
        
        # Build narrative response
        risk_emoji = "🚨" if prob >= 0.85 else "🔴" if prob >= 0.65 else "🟡" if prob >= 0.35 else "🟢"
        
        explanation = f"""## {risk_emoji} Churn Risk Analysis

**Status: {risk} Risk ({prob*100:.1f}% Churn Probability)**

### Why They Might Leave

"""
        
        # Add driver-specific context
        if drivers:
            for driver in drivers:
                context = driver_explanations.get(
                    driver,
                    f"{driver} is a significant churn indicator."
                )
                explanation += f"- **{driver}**: {context}\n"
        else:
            explanation += "- No major risk factors identified\n"
        
        explanation += f"""
### Recommended Actions This Week

1. **Reach Out**
   - Schedule success check-in (30 min)
   - Understand current challenges and priorities
   - Identify gaps between needs and product usage

2. **Create Quick Win**
   - Implement one requested feature
   - Resolve a top support pain point
   - Demonstrate tangible progress

### Business Impact

Customers at this risk level often become either loyal advocates or churn within 30-60 days. 
Proactive engagement now prevents revenue loss and builds long-term loyalty.

---
*Generated using rule-based analysis. For AI-powered insights, refresh the page.*
"""
        
        logger.info("📋 Using fallback explanation (rule-based)")
        return explanation.strip()


# Global service instance
genai_service = EnhancedGenAIService()


def generate_explanation(ds_output: Dict[str, Any]) -> str:
    """
    Wrapper function for backward compatibility
    Generates AI explanation with fallback support
    """
    return genai_service.generate_explanation(ds_output)
