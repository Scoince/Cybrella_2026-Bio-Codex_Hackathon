"""
Hospital Finder Agent for HealthGuard AI.

Recommends nearby hospitals/clinics based on the differential diagnosis,
using the Overpass API (OpenStreetMap) for hospital search and Nominatim
for address geocoding. Both APIs are free and require no authentication.
"""

from __future__ import annotations

import math
import re
import requests
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api"
USER_AGENT = "HealthGuardAI/1.0 (educational prototype)"

# ---------------------------------------------------------------------------
# Condition → Specialty Mapping
# ---------------------------------------------------------------------------

CONDITION_SPECIALTY_MAP: dict[str, dict] = {
    "Community-Acquired Pneumonia (CAP)": {
        "specialty": "Pulmonology",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "pneumology",
        },
        "urgency": "Urgent",
    },
    "Acute Heart Failure / Decompensated Heart Failure": {
        "specialty": "Cardiology",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "cardiology",
        },
        "urgency": "Emergency",
    },
    "Acute Coronary Syndrome (ACS)": {
        "specialty": "Cardiology (Interventional)",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "cardiology",
        },
        "urgency": "Emergency",
    },
    "COPD Exacerbation": {
        "specialty": "Pulmonology",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "pneumology",
        },
        "urgency": "Urgent",
    },
    "Asthma Exacerbation": {
        "specialty": "Pulmonology / Allergy",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "pneumology",
        },
        "urgency": "Urgent",
    },
    "Pulmonary Embolism (PE)": {
        "specialty": "Pulmonology / Vascular Medicine",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "pneumology",
        },
        "urgency": "Emergency",
    },
    "Sepsis": {
        "specialty": "Critical Care / Emergency Medicine",
        "osm_tags": {
            "amenity": "hospital",
            "emergency": "yes",
        },
        "urgency": "Emergency",
    },
    "Acute Ischemic Stroke": {
        "specialty": "Neurology (Stroke Center)",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "neurology",
        },
        "urgency": "Emergency",
    },
    "Type 2 Diabetes \u2013 Acute Complications": {
        "specialty": "Endocrinology",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "endocrinology",
        },
        "urgency": "Urgent",
    },
    "COVID-19": {
        "specialty": "Infectious Disease / Pulmonology",
        "osm_tags": {
            "amenity": "hospital",
            "healthcare:speciality": "infectious_diseases",
        },
        "urgency": "Urgent",
    },
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in kilometres
    using the Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Geocoding (Nominatim)
# ---------------------------------------------------------------------------

def geocode_address(address: str) -> tuple[float, float] | None:
    """
    Convert a text address / city name to (lat, lon) via Nominatim.
    Returns None if geocoding fails.
    """
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(NOMINATIM_URL, params=params,
                            headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        return None
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Location autocomplete (Photon API)
# ---------------------------------------------------------------------------

def _format_photon_suggestion(feature: dict) -> str:
    """
    Build a human-readable label from a Photon GeoJSON feature.

    Example output: "Singjamei, Imphal, Manipur, India (suburb)"
    """
    props = feature.get("properties", {})
    parts: list[str] = []

    name = props.get("name", "")
    if name:
        parts.append(name)

    for key in ("district", "city", "county", "state", "country"):
        val = props.get(key, "")
        if val and val not in parts:
            parts.append(val)

    label = ", ".join(parts) if parts else "Unknown location"

    osm_value = props.get("osm_value", "")
    place_type = props.get("type", osm_value)
    if place_type:
        label += f" ({place_type})"

    return label


def fetch_location_suggestions(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Fetch place suggestions from the Photon geocoding API.

    Returns a list of dicts with keys: label, lat, lon.
    Returns an empty list on error or if no results found.
    """
    if not query or len(query.strip()) < 3:
        return []

    params = {
        "q": query.strip(),
        "limit": limit,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(
            PHOTON_URL,
            params=params,
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    suggestions: list[dict] = []
    seen_labels: set[str] = set()

    for feature in data.get("features", []):
        coords = feature.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            continue

        label = _format_photon_suggestion(feature)

        if label in seen_labels:
            continue
        seen_labels.add(label)

        suggestions.append({
            "label": label,
            "lon": float(coords[0]),   # GeoJSON is [lon, lat]
            "lat": float(coords[1]),
        })

    return suggestions


# ---------------------------------------------------------------------------
# Hospital search (Overpass API)
# ---------------------------------------------------------------------------

def get_nearby_hospitals(
    lat: float,
    lon: float,
    radius_km: float = 10,
    specialty_tags: dict | None = None,
) -> list[dict]:
    """
    Query the Overpass API for hospitals and clinics near the given
    coordinates.

    Returns a list of dicts with keys:
        name, lat, lon, distance_km, type, address, phone,
        specialty, website, specialty_match, emergency
    """
    radius_m = int(radius_km * 1000)

    # Broad query: all hospitals + clinics within radius
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lon});
      way["amenity"="hospital"](around:{radius_m},{lat},{lon});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lon});
      node["amenity"="clinic"](around:{radius_m},{lat},{lon});
      way["amenity"="clinic"](around:{radius_m},{lat},{lon});
      node["healthcare"="hospital"](around:{radius_m},{lat},{lon});
      way["healthcare"="hospital"](around:{radius_m},{lat},{lon});
      node["healthcare"="clinic"](around:{radius_m},{lat},{lon});
      way["healthcare"="clinic"](around:{radius_m},{lat},{lon});
      node["healthcare"="doctor"](around:{radius_m},{lat},{lon});
      way["healthcare"="doctor"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """

    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    hospitals: list[dict] = []
    seen_names: set[str] = set()

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        # Coordinates: nodes have lat/lon directly; ways/relations use "center"
        h_lat = element.get("lat") or element.get("center", {}).get("lat")
        h_lon = element.get("lon") or element.get("center", {}).get("lon")
        if h_lat is None or h_lon is None:
            continue

        # Type
        amenity = tags.get("amenity", "")
        healthcare = tags.get("healthcare", "")
        if "hospital" in (amenity + " " + healthcare):
            h_type = "Hospital"
        elif "clinic" in (amenity + " " + healthcare):
            h_type = "Clinic"
        elif "doctor" in (amenity + " " + healthcare):
            h_type = "Doctor"
        else:
            h_type = "Medical Facility"

        # Specialty match
        h_specialty = tags.get("healthcare:speciality", "")
        specialty_match = False
        if specialty_tags and specialty_tags.get("healthcare:speciality"):
            target = specialty_tags["healthcare:speciality"].lower()
            specialty_match = target in h_specialty.lower()

        # Address from addr:* tags
        addr_parts = []
        for key in ["addr:housenumber", "addr:street", "addr:city",
                     "addr:postcode", "addr:state"]:
            if tags.get(key):
                addr_parts.append(tags[key])
        address = ", ".join(addr_parts) if addr_parts else tags.get("addr:full", "")

        distance = haversine_distance(lat, lon, h_lat, h_lon)

        hospitals.append({
            "name": name,
            "lat": h_lat,
            "lon": h_lon,
            "distance_km": round(distance, 2),
            "type": h_type,
            "address": address,
            "phone": tags.get("phone", tags.get("contact:phone", "")),
            "specialty": h_specialty,
            "website": tags.get("website", tags.get("contact:website", "")),
            "specialty_match": specialty_match,
            "emergency": tags.get("emergency", "") == "yes",
        })

    return hospitals


# ---------------------------------------------------------------------------
# Condition name parsing from differential text
# ---------------------------------------------------------------------------

def _parse_top_condition(differential_text: str) -> str | None:
    """
    Extract the #1 ranked condition name from the differential diagnosis
    markdown. Handles both rule-based and LLM output formats.
    """
    # Fallback format: "## 1. Community-Acquired Pneumonia (CAP)"
    match = re.search(r'##\s*1\.\s*(.+)', differential_text)
    if match:
        return match.group(1).strip()

    # LLM format: "1. **Condition Name**" or "1. Condition Name"
    match = re.search(r'(?:^|\n)\s*1[\.\)]\s*\*{0,2}([^*\n]+)', differential_text)
    if match:
        return match.group(1).strip()

    return None


def _match_condition_name(parsed_name: str) -> str | None:
    """
    Fuzzy-match a parsed condition name against CONDITION_SPECIALTY_MAP keys.
    Uses exact → substring → keyword cascade.
    """
    if not parsed_name:
        return None

    parsed_lower = parsed_name.lower()

    # Exact match
    for key in CONDITION_SPECIALTY_MAP:
        if key.lower() == parsed_lower:
            return key

    # Substring match
    for key in CONDITION_SPECIALTY_MAP:
        key_lower = key.lower()
        if key_lower in parsed_lower or parsed_lower in key_lower:
            return key

    # Keyword fallback
    KEYWORD_MAP = {
        "pneumonia": "Community-Acquired Pneumonia (CAP)",
        "heart failure": "Acute Heart Failure / Decompensated Heart Failure",
        "coronary": "Acute Coronary Syndrome (ACS)",
        "acs": "Acute Coronary Syndrome (ACS)",
        "myocardial infarction": "Acute Coronary Syndrome (ACS)",
        "stemi": "Acute Coronary Syndrome (ACS)",
        "nstemi": "Acute Coronary Syndrome (ACS)",
        "copd": "COPD Exacerbation",
        "chronic obstructive": "COPD Exacerbation",
        "asthma": "Asthma Exacerbation",
        "pulmonary embolism": "Pulmonary Embolism (PE)",
        "embolism": "Pulmonary Embolism (PE)",
        "sepsis": "Sepsis",
        "septic": "Sepsis",
        "stroke": "Acute Ischemic Stroke",
        "cerebrovascular": "Acute Ischemic Stroke",
        "diabetes": "Type 2 Diabetes \u2013 Acute Complications",
        "diabetic ketoacidosis": "Type 2 Diabetes \u2013 Acute Complications",
        "dka": "Type 2 Diabetes \u2013 Acute Complications",
        "hhs": "Type 2 Diabetes \u2013 Acute Complications",
        "covid": "COVID-19",
        "sars-cov": "COVID-19",
    }
    for keyword, condition in KEYWORD_MAP.items():
        if keyword in parsed_lower:
            return condition

    return None


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def recommend_hospitals(
    differential_text: str,
    lat: float,
    lon: float,
    radius_km: float = 10,
) -> dict:
    """
    Parse the top diagnosis, look up the appropriate medical specialty,
    query nearby hospitals, and sort by relevance.

    Returns dict with keys:
        condition, specialty, urgency, user_location,
        hospitals (list), error (str|None)
    """
    result: dict[str, Any] = {
        "condition": None,
        "specialty": "General Medicine",
        "urgency": "Routine",
        "user_location": (lat, lon),
        "hospitals": [],
        "error": None,
    }

    # 1. Parse top condition from differential
    parsed = _parse_top_condition(differential_text)
    condition = _match_condition_name(parsed) if parsed else None

    # 2. Look up specialty
    specialty_tags = None
    if condition and condition in CONDITION_SPECIALTY_MAP:
        info = CONDITION_SPECIALTY_MAP[condition]
        result["condition"] = condition
        result["specialty"] = info["specialty"]
        result["urgency"] = info["urgency"]
        specialty_tags = info["osm_tags"]
    else:
        result["condition"] = parsed or "Unknown"

    # 3. Query hospitals
    hospitals = get_nearby_hospitals(lat, lon, radius_km, specialty_tags)

    if not hospitals:
        result["error"] = (
            f"No hospitals or clinics found within {radius_km} km. "
            "Try increasing the search radius or entering a different location."
        )
        return result

    # 4. Sort: specialty matches first → emergency depts → distance
    def sort_key(h):
        return (
            not h["specialty_match"],
            not h["emergency"],
            h["distance_km"],
        )

    hospitals.sort(key=sort_key)
    result["hospitals"] = hospitals

    return result
