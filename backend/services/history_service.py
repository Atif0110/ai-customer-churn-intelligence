"""
Customer analysis history storage service
Persistent storage with error handling and validation
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
MAX_HISTORY_RECORDS = 10000  # Prevent file from growing too large


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_data_directory() -> None:
    """
    Ensure data directory exists
    """
    try:
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        logger.debug(f"✅ Data directory ready: {DATA_DIR}")
    except Exception as e:
        logger.error(f"❌ Failed to create data directory: {e}")
        raise


def ensure_history_file() -> None:
    """
    Ensure history file exists
    Creates empty array if missing
    """
    ensure_data_directory()
    
    try:
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as f:
                json.dump([], f)
            logger.info(f"✅ Created history file: {HISTORY_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to create history file: {e}")
        raise


def load_history() -> List[Dict[str, Any]]:
    """
    Load history from file with error handling
    
    Returns:
        List of history records, empty list on error
    """
    ensure_history_file()
    
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning("⚠️ History file corrupted, resetting")
            return []
        
        logger.debug(f"📖 Loaded {len(data)} history records")
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ History file corrupted: {e}. Resetting.")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading history: {e}")
        return []


def save_history(data: List[Dict[str, Any]]) -> bool:
    """
    Save history to file with error handling
    
    Args:
        data: List of records to save
    
    Returns:
        True if successful, False otherwise
    """
    ensure_data_directory()
    
    try:
        # Backup existing file before overwriting
        if os.path.exists(HISTORY_FILE):
            backup_file = HISTORY_FILE + ".bak"
            try:
                with open(HISTORY_FILE, "r") as f:
                    backup_data = json.load(f)
                with open(backup_file, "w") as f:
                    json.dump(backup_data, f)
                logger.debug(f"📦 Backup created: {backup_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create backup: {e}")
        
        # Write new data
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved {len(data)} records to history")
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to save history: {e}")
        return False


# ============================================================================
# HISTORY SERVICE FUNCTIONS
# ============================================================================

def save_record(
    values: List[float],
    result: Dict[str, Any]
) -> bool:
    """
    Save a prediction record to history
    
    Args:
        values: Input values [v1, v2, v3]
        result: Prediction result dict
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing history
        history = load_history()
        
        # Validate result structure
        if not isinstance(result, dict):
            logger.error("❌ Invalid result structure")
            return False
        
        # Create record
        record = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "usage_hours": float(values[0]),
                "support_tickets": float(values[1]),
                "tenure_months": float(values[2])
            },
            "output": {
                "churn_probability": float(result.get("churn_probability", 0)),
                "risk_level": result.get("risk_level", "Unknown"),
                "drivers": result.get("drivers", []),
                "confidence": float(result.get("confidence_score", 0))
            }
        }
        
        # Append record
        history.append(record)
        
        # Limit history size (keep newest)
        if len(history) > MAX_HISTORY_RECORDS:
            history = history[-MAX_HISTORY_RECORDS:]
            logger.warning(
                f"⚠️ History truncated to {MAX_HISTORY_RECORDS} most recent records"
            )
        
        # Save to file
        if save_history(history):
            logger.info(f"✅ Recorded analysis (total: {len(history)})")
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"❌ Error saving record: {e}")
        return False


def get_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get analysis history
    
    Args:
        limit: Maximum number of records to return (newest first)
    
    Returns:
        List of history records
    """
    try:
        history = load_history()
        
        # Return newest first
        history = list(reversed(history))
        
        # Limit if requested
        if limit and len(history) > limit:
            history = history[:limit]
        
        logger.info(f"📖 Retrieved {len(history)} history records")
        return history
    
    except Exception as e:
        logger.error(f"❌ Error getting history: {e}")
        return []


def get_statistics() -> Dict[str, Any]:
    """
    Get statistics about analysis history
    
    Returns:
        Statistics dict with counts and averages
    """
    try:
        history = load_history()
        
        if not history:
            return {
                "total_analyses": 0,
                "message": "No analysis history yet"
            }
        
        # Extract predictions
        predictions = [
            float(r.get("output", {}).get("churn_probability", 0))
            for r in history if "output" in r
        ]
        
        # Calculate stats
        stats = {
            "total_analyses": len(history),
            "average_churn_probability": round(sum(predictions) / len(predictions), 3) if predictions else 0,
            "min_churn": round(min(predictions), 3) if predictions else 0,
            "max_churn": round(max(predictions), 3) if predictions else 0,
            "latest_timestamp": history[-1].get("timestamp") if history else None
        }
        
        logger.info(f"📊 History statistics: {len(history)} total analyses")
        return stats
    
    except Exception as e:
        logger.error(f"❌ Error calculating statistics: {e}")
        return {"error": str(e)}


def clear_history() -> bool:
    """
    Clear all history (for testing/reset)
    
    Returns:
        True if successful
    """
    try:
        if save_history([]):
            logger.warning("⚠️ History cleared")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        return False


# Initialize on import
try:
    ensure_history_file()
except Exception as e:
    logger.warning(f"⚠️ Could not initialize history file: {e}")
