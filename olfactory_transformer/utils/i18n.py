"""Internationalization support for Olfactory Transformer."""

import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguages(Enum):
    """Supported languages for multi-lingual scent descriptions."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    KOREAN = "ko"


class ScentDescriptorTranslator:
    """Multi-lingual scent descriptor translator."""
    
    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.translations = self._load_translations()
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load scent descriptor translations."""
        # Common scent descriptors in multiple languages
        translations = {
            "en": {
                "floral": "floral",
                "fresh": "fresh", 
                "sweet": "sweet",
                "woody": "woody",
                "citrus": "citrus",
                "spicy": "spicy",
                "herbal": "herbal",
                "marine": "marine",
                "fruity": "fruity",
                "earthy": "earthy",
                "musky": "musky",
                "powdery": "powdery",
                "green": "green",
                "animalic": "animalic",
                "smoky": "smoky",
                "metallic": "metallic",
                "minty": "minty",
                "vanilla": "vanilla",
                "rose": "rose",
                "lavender": "lavender",
                "jasmine": "jasmine",
                "sandalwood": "sandalwood",
                "cedar": "cedar",
                "amber": "amber",
                "bergamot": "bergamot",
                "lemon": "lemon",
                "orange": "orange",
                "peppermint": "peppermint",
                "eucalyptus": "eucalyptus",
            },
            "es": {
                "floral": "floral",
                "fresh": "fresco",
                "sweet": "dulce", 
                "woody": "amaderado",
                "citrus": "cítrico",
                "spicy": "especiado",
                "herbal": "herbal",
                "marine": "marino",
                "fruity": "afrutado",
                "earthy": "terroso",
                "musky": "almizclado",
                "powdery": "empolvado",
                "green": "verde",
                "animalic": "animalico",
                "smoky": "ahumado",
                "metallic": "metálico",
                "minty": "mentolado",
                "vanilla": "vainilla",
                "rose": "rosa",
                "lavender": "lavanda",
                "jasmine": "jazmín",
                "sandalwood": "sándalo",
                "cedar": "cedro",
                "amber": "ámbar",
                "bergamot": "bergamota",
                "lemon": "limón",
                "orange": "naranja",
                "peppermint": "menta",
                "eucalyptus": "eucalipto",
            },
            "fr": {
                "floral": "floral",
                "fresh": "frais",
                "sweet": "sucré",
                "woody": "boisé",
                "citrus": "agrume", 
                "spicy": "épicé",
                "herbal": "herbal",
                "marine": "marin",
                "fruity": "fruité",
                "earthy": "terreux",
                "musky": "musqué",
                "powdery": "poudré",
                "green": "vert",
                "animalic": "animalique",
                "smoky": "fumé",
                "metallic": "métallique",
                "minty": "mentholé",
                "vanilla": "vanille",
                "rose": "rose",
                "lavender": "lavande",
                "jasmine": "jasmin",
                "sandalwood": "santal",
                "cedar": "cèdre",
                "amber": "ambre",
                "bergamot": "bergamote",
                "lemon": "citron",
                "orange": "orange",
                "peppermint": "menthe poivrée",
                "eucalyptus": "eucalyptus",
            },
            "de": {
                "floral": "blumig",
                "fresh": "frisch",
                "sweet": "süß",
                "woody": "holzig",
                "citrus": "zitrusartig",
                "spicy": "würzig",
                "herbal": "kräuterig",
                "marine": "marin",
                "fruity": "fruchtig",
                "earthy": "erdig",
                "musky": "moschusartig",
                "powdery": "pudrig",
                "green": "grün",
                "animalic": "animalisch",
                "smoky": "rauchig",
                "metallic": "metallisch",
                "minty": "minzig",
                "vanilla": "vanille",
                "rose": "rose",
                "lavender": "lavendel",
                "jasmine": "jasmin",
                "sandalwood": "sandelholz",
                "cedar": "zeder",
                "amber": "amber",
                "bergamot": "bergamotte",
                "lemon": "zitrone",
                "orange": "orange",
                "peppermint": "pfefferminze",
                "eucalyptus": "eukalyptus",
            },
            "ja": {
                "floral": "花の香り",
                "fresh": "爽やか",
                "sweet": "甘い",
                "woody": "ウッディ",
                "citrus": "シトラス",
                "spicy": "スパイシー",
                "herbal": "ハーバル",
                "marine": "マリン",
                "fruity": "フルーティ",
                "earthy": "アーシー",
                "musky": "ムスキー",
                "powdery": "パウダリー",
                "green": "グリーン",
                "animalic": "アニマリック",
                "smoky": "スモーキー",
                "metallic": "メタリック",
                "minty": "ミント",
                "vanilla": "バニラ",
                "rose": "バラ",
                "lavender": "ラベンダー",
                "jasmine": "ジャスミン",
                "sandalwood": "サンダルウッド",
                "cedar": "シダー",
                "amber": "アンバー",
                "bergamot": "ベルガモット",
                "lemon": "レモン",
                "orange": "オレンジ",
                "peppermint": "ペパーミント",
                "eucalyptus": "ユーカリ",
            },
            "zh": {
                "floral": "花香",
                "fresh": "清新",
                "sweet": "甜美",
                "woody": "木质",
                "citrus": "柑橘",
                "spicy": "辛香",
                "herbal": "草本",
                "marine": "海洋",
                "fruity": "果香",
                "earthy": "泥土",
                "musky": "麝香",
                "powdery": "粉质",
                "green": "绿色",
                "animalic": "动物",
                "smoky": "烟熏",
                "metallic": "金属",
                "minty": "薄荷",
                "vanilla": "香草",
                "rose": "玫瑰",
                "lavender": "薰衣草",
                "jasmine": "茉莉",
                "sandalwood": "檀香",
                "cedar": "雪松",
                "amber": "琥珀",
                "bergamot": "佛手柑",
                "lemon": "柠檬",
                "orange": "橙子",
                "peppermint": "胡椒薄荷",
                "eucalyptus": "桉树",
            }
        }
        
        return translations
        
    def translate_descriptor(self, descriptor: str, target_language: str) -> str:
        """Translate a scent descriptor to target language."""
        if target_language not in self.translations:
            logger.warning(f"Language {target_language} not supported, using English")
            return descriptor
            
        # Find English key for the descriptor
        english_key = None
        for key, value in self.translations["en"].items():
            if value.lower() == descriptor.lower():
                english_key = key
                break
                
        if not english_key:
            # If not found, try as key directly
            english_key = descriptor.lower()
            
        # Return translation or original if not found
        return self.translations[target_language].get(english_key, descriptor)
        
    def translate_descriptors(self, descriptors: List[str], target_language: str) -> List[str]:
        """Translate a list of scent descriptors."""
        return [self.translate_descriptor(desc, target_language) for desc in descriptors]
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.translations.keys())
        
    def detect_language(self, text: str) -> str:
        """Simple language detection for scent descriptors."""
        # Count matches for each language
        scores = {}
        for lang, translations in self.translations.items():
            score = sum(1 for word in text.lower().split() 
                       if word in translations.values())
            scores[lang] = score
            
        # Return language with highest score, default to English
        if scores:
            return max(scores, key=scores.get)
        return "en"


class RegionalComplianceManager:
    """Manage regional compliance for fragrance regulations."""
    
    def __init__(self):
        self.regulations = self._load_regulations()
        
    def _load_regulations(self) -> Dict[str, Dict[str, any]]:
        """Load regional fragrance regulations."""
        return {
            "EU": {
                "name": "European Union REACH/CLP",
                "restricted_substances": [
                    "atranol", "chloroatranol", "HICC", "BMHCA"
                ],
                "labeling_required": [
                    "linalool", "limonene", "citronellol", "geraniol",
                    "cinnamyl alcohol", "benzyl alcohol", "citral"
                ],
                "concentration_limits": {
                    "linalool": 0.001,  # 0.1%
                    "limonene": 0.001,
                }
            },
            "US": {
                "name": "United States FDA/CPSC",
                "restricted_substances": [
                    "methyl eugenol", "safrole"
                ],
                "labeling_required": [],
                "concentration_limits": {}
            },
            "IFRA": {
                "name": "International Fragrance Association",
                "categories": {
                    1: "Products applied to lips",
                    2: "Products applied to axillae", 
                    3: "Products applied to face/hands",
                    4: "Fine fragrance products",
                    5: "Products applied to body",
                    6: "Products with dilute application",
                    7: "Products applied to hair",
                    8: "Products with body/room sprays",
                    9: "Products with limited body exposure",
                    10: "Products with very limited exposure",
                    11: "Products not applied to skin"
                },
                "restrictions": {
                    "oakmoss": {1: 0, 2: 0, 3: 0, 4: 0.1},
                    "treemoss": {1: 0, 2: 0, 3: 0, 4: 0.1}
                }
            }
        }
        
    def check_compliance(self, ingredients: List[str], region: str = "EU") -> Dict[str, any]:
        """Check ingredient compliance for a region."""
        if region not in self.regulations:
            raise ValueError(f"Region {region} not supported")
            
        regulation = self.regulations[region]
        compliance_report = {
            "compliant": True,
            "warnings": [],
            "restrictions": [],
            "labeling_required": []
        }
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            
            # Check restrictions
            if ingredient_lower in [s.lower() for s in regulation.get("restricted_substances", [])]:
                compliance_report["compliant"] = False
                compliance_report["restrictions"].append(f"Banned substance: {ingredient}")
                
            # Check labeling requirements
            if ingredient_lower in [s.lower() for s in regulation.get("labeling_required", [])]:
                compliance_report["labeling_required"].append(ingredient)
                
        return compliance_report
        
    def get_ifra_category_limits(self, ingredient: str, category: int) -> Optional[float]:
        """Get IFRA concentration limits for ingredient in category."""
        ifra = self.regulations["IFRA"]
        ingredient_lower = ingredient.lower()
        
        if ingredient_lower in ifra["restrictions"]:
            return ifra["restrictions"][ingredient_lower].get(category)
        return None


class MultiRegionManager:
    """Multi-region deployment and data management."""
    
    def __init__(self):
        self.regions = {
            "us-east-1": {"name": "US East", "compliance": ["US", "IFRA"]},
            "us-west-2": {"name": "US West", "compliance": ["US", "IFRA"]},
            "eu-west-1": {"name": "EU Ireland", "compliance": ["EU", "IFRA"]},
            "eu-central-1": {"name": "EU Germany", "compliance": ["EU", "IFRA"]},
            "ap-northeast-1": {"name": "Asia Pacific Tokyo", "compliance": ["IFRA"]},
            "ap-southeast-1": {"name": "Asia Pacific Singapore", "compliance": ["IFRA"]},
        }
        
    def get_optimal_region(self, user_location: str) -> str:
        """Get optimal region for user location."""
        location_mapping = {
            "US": "us-east-1",
            "CA": "us-east-1", 
            "MX": "us-east-1",
            "GB": "eu-west-1",
            "FR": "eu-west-1",
            "DE": "eu-central-1",
            "IT": "eu-west-1",
            "ES": "eu-west-1",
            "JP": "ap-northeast-1",
            "KR": "ap-northeast-1",
            "SG": "ap-southeast-1",
            "AU": "ap-southeast-1",
        }
        
        return location_mapping.get(user_location.upper(), "us-east-1")
        
    def get_compliance_requirements(self, region: str) -> List[str]:
        """Get compliance requirements for a region."""
        if region in self.regions:
            return self.regions[region]["compliance"]
        return ["IFRA"]  # Default fallback


# Global instances
translator = ScentDescriptorTranslator()
compliance_manager = RegionalComplianceManager()
region_manager = MultiRegionManager()