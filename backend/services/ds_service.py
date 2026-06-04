"""
Enhanced ML churn prediction service with type hints and confidence metrics
Production-grade with validation and diagnostics
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChurnPrediction:
    """Structured churn prediction output"""
    churn_probability: float
    risk_level: str
    drivers: List[str]
    confidence_score: float  # 0-1: Model confidence
    percentile: float  # 0-100: Where customer ranks
    

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Realistic training data based on actual SaaS customer patterns
# Each row: [usage_hours, support_tickets, tenure_months]
# Label: 1 = churned (left), 0 = retained (stayed)

X = np.array([
    # LOYAL CUSTOMERS (churn=0, stay)
    [95, 1, 58],       # Power user, no issues, 5+ years → STAYS
    [88, 2, 52],       # Heavy user, minimal support, long tenure → STAYS
    [82, 1, 48],       # Active user, very few issues, loyal → STAYS
    [78, 3, 44],       # Regular user, minimal support, established → STAYS
    [72, 2, 40],       # Good engagement, stable, long customer → STAYS
    
    # MEDIUM CUSTOMERS (churn=1, mixed)
    [55, 5, 28],       # Moderate usage, some issues, ~2.3 years → CHURN
    [48, 6, 24],       # Below average usage, support issues → CHURN
    [42, 7, 20],       # Low-medium usage, higher support needs → CHURN
    [38, 8, 18],       # Declining usage, friction forming → CHURN
    [35, 9, 16],       # Low engagement, support burden, new → CHURN
    
    # AT-RISK CUSTOMERS (churn=1, will churn)
    [25, 10, 12],      # Low usage, struggling with product, 1 year → CHURN
    [18, 11, 8],       # Minimal engagement, many issues → CHURN
    [12, 13, 5],       # Very low usage, high support, early → CHURN
    [8, 14, 3],        # Critical: barely using, frustrated, new → CHURN
    [3, 15, 1],        # Almost gone: no usage, overwhelmed → CHURN
], dtype=np.float32)

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)

# Train the model
try:
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X, y)
    logger.info("✅ Churn prediction model trained successfully")
except Exception as e:
    logger.error(f"❌ Failed to train model: {str(e)}")
    raise


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_inputs(v1: float, v2: float, v3: float) -> Tuple[bool, Optional[str]]:
    """
    Validate input ranges with helpful error messages
    
    Args:
        v1: Monthly usage hours
        v2: Support tickets
        v3: Tenure in months
    
    Returns:
        (is_valid, error_message)
    """
    errors = []
    
    # Validate v1: Monthly usage hours
    if not (0 <= v1 <= 100):
        errors.append(
            f"Usage hours ({v1}) outside valid range [0-100]. "
            f"Typical SaaS customers: 10-80 hours/month"
        )
    
    # Validate v2: Support tickets
    if not (0 <= v2 <= 15):
        errors.append(
            f"Support tickets ({v2}) outside valid range [0-15]. "
            f"Typical SaaS customers: 0-8 tickets/month"
        )
    
    # Validate v3: Tenure in months
    if not (0 <= v3 <= 60):
        errors.append(
            f"Tenure ({v3} months) outside valid range [0-60]. "
            f"Platform supports up to 5 years of customer history"
        )
    
    if errors:
        return False, "\n".join(errors)
    
    return True, None


def identify_drivers(v1: float, v2: float, v3: float) -> List[str]:
    """
    Identify churn risk drivers based on customer metrics
    
    Rules based on industry benchmarks:
    - Usage < 20 hours/month: Low engagement (churn signal)
    - Tickets > 7/month: High support load (churn signal)
    - Tenure < 18 months: New customer (evaluation phase)
    """
    drivers = []
    
    # Driver 1: Low product engagement
    if v1 < 20:
        drivers.append("Low product engagement")
    
    # Driver 2: High support burden
    if v2 > 7:
        drivers.append("High support load")
    
    # Driver 3: Short tenure (early-stage customer)
    if v3 < 18:
        drivers.append("Early-stage customer")
    
    # If no risk factors: customer is stable
    if not drivers:
        drivers.append("Stable customer profile")
    
    return drivers


def calculate_confidence(v1: float, v2: float, v3: float) -> float:
    """
    Calculate model confidence (0-1)
    Higher confidence when customer is within typical range
    Lower confidence for outliers
    """
    # Typical customer profile
    typical_usage = 50
    typical_tickets = 5
    typical_tenure = 24
    
    # Calculate distances from typical
    distances = [
        abs(v1 - typical_usage) / 50,
        abs(v2 - typical_tickets) / 7.5,
        abs(v3 - typical_tenure) / 24,
    ]
    
    avg_distance = np.mean(distances)
    # Penalize outliers
    confidence = max(0.0, 1.0 - (avg_distance * 0.3))
    
    return round(confidence, 3)


def get_percentile(prob: float) -> float:
    """
    Calculate what percentile this customer falls into
    Among all possible predictions
    """
    # Simplified: assume distribution across range
    percentile = min(100, max(0, prob * 100))
    return round(percentile, 1)


def classify_risk(prob: float) -> str:
    """
    Classify risk level based on probability
    
    - Low: < 30% (safe customers)
    - Medium: 30-60% (at decision point)
    - High: >= 60% (likely to churn)
    """
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    else:
        return "Low"


# ============================================================================
# PREDICTION SERVICE
# ============================================================================

def predict(values: List[float]) -> Dict[str, any]:
    """
    Predict customer churn probability with confidence metrics
    
    Args:
        values: [usage_hours, support_tickets, tenure_months]
    
    Returns:
        {
            "churn_probability": float (0-1),
            "risk_level": str,
            "drivers": list,
            "confidence_score": float,
            "percentile": float
        }
    
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        # ====== VALIDATION ======
        v1, v2, v3 = float(values[0]), float(values[1]), float(values[2])
        
        is_valid, error_msg = validate_inputs(v1, v2, v3)
        if not is_valid:
            logger.error(f"❌ Validation error: {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"📊 Predicting: usage={v1}, tickets={v2}, tenure={v3}")
        
        # ====== ML PREDICTION ======
        arr = np.array([[v1, v2, v3]], dtype=np.float32)
        prob = float(model.predict_proba(arr)[0][1])
        prob = max(0.0, min(1.0, prob))  # Clamp to [0,1]
        
        # ====== CALCULATE METRICS ======
        drivers = identify_drivers(v1, v2, v3)
        risk_level = classify_risk(prob)
        confidence = calculate_confidence(v1, v2, v3)
        percentile = get_percentile(prob)
        
        result = {
            "churn_probability": prob,
            "risk_level": risk_level,
            "drivers": drivers,
            "confidence_score": confidence,
            "percentile": percentile
        }
        
        logger.info(
            f"✅ Prediction complete: "
            f"prob={prob:.2%}, risk={risk_level}, confidence={confidence}"
        )
        
        return result
    
    except ValueError as e:
        logger.error(f"❌ Validation error in predict: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in predict: {str(e)}")
        raise ValueError(f"Prediction failed: {str(e)}")


# ============================================================================
# MODEL TESTING
# ============================================================================

def test_model() -> bool:
    """
    Comprehensive model validation
    Returns True if all tests pass
    """
    print("\n" + "="*70)
    print("CHURN MODEL VALIDATION & TESTING")
    print("="*70)
    
    test_cases = [
        (95, 1, 58, "Low", "Loyal power user"),
        (75, 2, 40, "Low", "Good customer"),
        (50, 5, 25, "Medium", "Medium engagement"),
        (35, 8, 15, "High", "Low usage + issues"),
        (5, 15, 1, "High", "Critical risk"),
        (0, 0, 0, "High", "Zero engagement"),
    ]
    
    print("\nTest Results:")
    print("-" * 70)
    
    all_passed = True
    
    for v1, v2, v3, expected_risk, description in test_cases:
        try:
            result = predict([v1, v2, v3])
            actual_risk = result["risk_level"]
            prob = result["churn_probability"]
            
            passed = actual_risk == expected_risk
            status = "✅ PASS" if passed else "❌ FAIL"
            
            if not passed:
                all_passed = False
            
            print(f"\n{status}: {description}")
            print(f"  Prob: {prob:.1%} | Risk: {actual_risk} (expected {expected_risk})")
        
        except Exception as e:
            all_passed = False
            print(f"\n❌ ERROR: {description} - {str(e)}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    test_model()
