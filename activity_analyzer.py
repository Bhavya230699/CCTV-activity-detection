"""Tactical activity classification for tracked objects."""

def classify_activity(object_type: str, moving: bool, speed: float = 0.0, threat_category: str | None = None) -> str:
    if threat_category:
        return f"CRITICAL: {threat_category} ACTIVITY"
    if not moving:
        return "Stationary / Loitering" if object_type == "Person" else "Stationary"
    
    if object_type == "Person":
        if speed > 180.0:
            return "Running / Rapid Movement"
        elif speed > 40.0:
            return "Walking / Moving"
        return "Slow Movement"
    elif object_type == "Animal":
        return "Animal Moving"
    elif object_type == "Vehicle":
        return "Vehicle Moving"
    return "Object in Motion"

def analyze_activity(object_type: str, moving: bool, speed: float = 0.0, threat_category: str | None = None) -> str:
    return classify_activity(object_type, moving, speed, threat_category)