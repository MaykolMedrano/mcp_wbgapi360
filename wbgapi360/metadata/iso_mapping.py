"""
ISO 3166-1 Alpha-3 Country Code Mapping
Complete mapping of 3-letter country codes to human-readable names.
Used for labels=True feature in wbgapi360.
"""

# Primary mapping: Code → Name
ISO_3166_NAMES = {
    # Americas
    "USA": "United States",
    "CAN": "Canada",
    "MEX": "Mexico",
    "BRA": "Brazil",
    "ARG": "Argentina",
    "CHL": "Chile",
    "COL": "Colombia",
    "PER": "Peru",
    "VEN": "Venezuela",
    "ECU": "Ecuador",
    "BOL": "Bolivia",
    "PRY": "Paraguay",
    "URY": "Uruguay",
    "CRI": "Costa Rica",
    "PAN": "Panama",
    "GTM": "Guatemala",
    "HND": "Honduras",
    "SLV": "El Salvador",
    "NIC": "Nicaragua",
    "CUB": "Cuba",
    "DOM": "Dominican Republic",
    "HTI": "Haiti",
    "JAM": "Jamaica",
    "TTO": "Trinidad and Tobago",
    
    # Europe
    "DEU": "Germany",
    "FRA": "France",
    "GBR": "United Kingdom",
    "ITA": "Italy",
    "ESP": "Spain",
    "POL": "Poland",
    "ROU": "Romania",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "CZE": "Czech Republic",
    "GRC": "Greece",
    "PRT": "Portugal",
    "SWE": "Sweden",
    "HUN": "Hungary",
    "AUT": "Austria",
    "BGR": "Bulgaria",
    "DNK": "Denmark",
    "FIN": "Finland",
    "SVK": "Slovakia",
    "NOR": "Norway",
    "IRL": "Ireland",
    "HRV": "Croatia",
    "LTU": "Lithuania",
    "SVN": "Slovenia",
    "LVA": "Latvia",
    "EST": "Estonia",
    "LUX": "Luxembourg",
    "MLT": "Malta",
    "ISL": "Iceland",
    "CHE": "Switzerland",
    "UKR": "Ukraine",
    "RUS": "Russia",
    "BLR": "Belarus",
    
    # Asia
    "CHN": "China",
    "IND": "India",
    "IDN": "Indonesia",
    "PAK": "Pakistan",
    "BGD": "Bangladesh",
    "JPN": "Japan",
    "PHL": "Philippines",
    "VNM": "Vietnam",
    "TUR": "Turkey",
    "IRN": "Iran",
    "THA": "Thailand",
    "MMR": "Myanmar",
    "KOR": "South Korea",
    "IRQ": "Iraq",
    "AFG": "Afghanistan",
    "SAU": "Saudi Arabia",
    "UZB": "Uzbekistan",
    "MYS": "Malaysia",
    "NPL": "Nepal",
    "YEM": "Yemen",
    "PRK": "North Korea",
    "LKA": "Sri Lanka",
    "KHM": "Cambodia",
    "JOR": "Jordan",
    "AZE": "Azerbaijan",
    "ARE": "United Arab Emirates",
    "TJK": "Tajikistan",
    "ISR": "Israel",
    "LAO": "Laos",
    "LBN": "Lebanon",
    "SGP": "Singapore",
    "OMN": "Oman",
    "KWT": "Kuwait",
    "GEO": "Georgia",
    "MNG": "Mongolia",
    "ARM": "Armenia",
    "QAT": "Qatar",
    "BHR": "Bahrain",
    "PSE": "Palestine",
    "KGZ": "Kyrgyzstan",
    "TKM": "Turkmenistan",
    "BRN": "Brunei",
    "MDV": "Maldives",
    "BTN": "Bhutan",
    
    # Africa
    "NGA": "Nigeria",
    "ETH": "Ethiopia",
    "EGY": "Egypt",
    "COD": "Democratic Republic of the Congo",
    "TZA": "Tanzania",
    "ZAF": "South Africa",
    "KEN": "Kenya",
    "UGA": "Uganda",
    "DZA": "Algeria",
    "SDN": "Sudan",
    "MAR": "Morocco",
    "GHA": "Ghana",
    "MOZ": "Mozambique",
    "MDG": "Madagascar",
    "CMR": "Cameroon",
    "CIV": "Ivory Coast",
    "NER": "Niger",
    "BFA": "Burkina Faso",
    "MLI": "Mali",
    "MWI": "Malawi",
    "ZMB": "Zambia",
    "SOM": "Somalia",
    "SEN": "Senegal",
    "TCD": "Chad",
    "ZWE": "Zimbabwe",
    "GIN": "Guinea",
    "RWA": "Rwanda",
    "BDI": "Burundi",
    "TUN": "Tunisia",
    "BEN": "Benin",
    "SSD": "South Sudan",
    "TGO": "Togo",
    "SLE": "Sierra Leone",
    "LBY": "Libya",
    "LBR": "Liberia",
    "MRT": "Mauritania",
    "CAF": "Central African Republic",
    "ERI": "Eritrea",
    "GMB": "Gambia",
    "BWA": "Botswana",
    "NAM": "Namibia",
    "GAB": "Gabon",
    "LSO": "Lesotho",
    "GNB": "Guinea-Bissau",
    "GNQ": "Equatorial Guinea",
    "MUS": "Mauritius",
    "SWZ": "Eswatini",
    "DJI": "Djibouti",
    "COM": "Comoros",
    "CPV": "Cape Verde",
    "STP": "Sao Tome and Principe",
    "SYC": "Seychelles",
    
    # Oceania
    "AUS": "Australia",
    "PNG": "Papua New Guinea",
    "NZL": "New Zealand",
    "FJI": "Fiji",
    "SLB": "Solomon Islands",
    "VUT": "Vanuatu",
    "WSM": "Samoa",
    "KIR": "Kiribati",
    "TON": "Tonga",
    "FSM": "Micronesia",
    "PLW": "Palau",
    "MHL": "Marshall Islands",
    "TUV": "Tuvalu",
    "NRU": "Nauru",
    
    # Special Codes (World Bank Aggregates)
    "WLD": "World",
    "EAS": "East Asia & Pacific",
    "ECS": "Europe & Central Asia",
    "LCN": "Latin America & Caribbean",
    "MEA": "Middle East & North Africa",
    "NAC": "North America",
    "SAS": "South Asia",
    "SSF": "Sub-Saharan Africa",
    "HIC": "High income",
    "UMC": "Upper middle income",
    "LMC": "Lower middle income",
    "LIC": "Low income",
}

# Reverse mapping: Name → Code (for reverse lookups)
ISO_3166_CODES = {v: k for k, v in ISO_3166_NAMES.items()}

def get_name(code: str) -> str:
    """
    Get human-readable name for ISO 3166-1 Alpha-3 code.
    Returns the original code if not found.
    
    Args:
        code: 3-letter ISO country code (e.g., "USA")
        
    Returns:
        Human-readable name (e.g., "United States") or original code if not found
    """
    return ISO_3166_NAMES.get(code.upper(), code)

def get_code(name: str) -> str:
    """
    Get ISO 3166-1 Alpha-3 code for country name.
    Returns the original name if not found.
    
    Args:
        name: Human-readable country name (e.g., "United States")
        
    Returns:
        3-letter ISO code (e.g., "USA") or original name if not found
    """
    return ISO_3166_CODES.get(name, name)
