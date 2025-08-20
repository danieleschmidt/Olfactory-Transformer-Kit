"""Enhanced I18n Manager with robust fallbacks and caching."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

# Language mappings with fallbacks
LANGUAGE_MAPPINGS = {
    'en': 'en-US',
    'es': 'es-ES', 
    'fr': 'fr-FR',
    'de': 'de-DE',
    'ja': 'ja-JP',
    'zh': 'zh-CN',
    'ko': 'ko-KR',
    'pt': 'pt-BR',
    'it': 'it-IT',
    'ru': 'ru-RU'
}

# Default translations for common terms
DEFAULT_TRANSLATIONS = {
    'en': {
        'model.loading': 'Loading model...',
        'model.ready': 'Model ready',
        'model.error': 'Model error',
        'prediction.processing': 'Processing prediction...',
        'prediction.complete': 'Prediction complete',
        'prediction.failed': 'Prediction failed',
        'error.invalid_smiles': 'Invalid SMILES format',
        'error.timeout': 'Request timeout',
        'error.validation_failed': 'Input validation failed',
        'error.processing_failed': 'Processing failed',
        'scent.floral': 'Floral',
        'scent.citrus': 'Citrus',
        'scent.woody': 'Woody',
        'scent.fresh': 'Fresh',
        'scent.sweet': 'Sweet',
        'scent.spicy': 'Spicy',
        'scent.herbal': 'Herbal',
        'scent.fruity': 'Fruity',
        'scent.green': 'Green',
        'scent.marine': 'Marine',
        'scent.musky': 'Musky',
        'scent.amber': 'Amber',
        'chemical_family.ester': 'Ester',
        'chemical_family.terpene': 'Terpene',
        'chemical_family.aldehyde': 'Aldehyde',
        'chemical_family.ketone': 'Ketone',
        'chemical_family.alcohol': 'Alcohol',
        'intensity.low': 'Low',
        'intensity.medium': 'Medium', 
        'intensity.high': 'High',
        'status.healthy': 'Healthy',
        'status.degraded': 'Degraded',
        'status.critical': 'Critical'
    },
    'es': {
        'model.loading': 'Cargando modelo...',
        'model.ready': 'Modelo listo',
        'model.error': 'Error del modelo',
        'prediction.processing': 'Procesando predicción...',
        'prediction.complete': 'Predicción completa',
        'prediction.failed': 'Predicción falló',
        'error.invalid_smiles': 'Formato SMILES inválido',
        'error.timeout': 'Tiempo de espera agotado',
        'error.validation_failed': 'Validación de entrada falló',
        'error.processing_failed': 'Procesamiento falló',
        'scent.floral': 'Floral',
        'scent.citrus': 'Cítrico',
        'scent.woody': 'Amaderado',
        'scent.fresh': 'Fresco',
        'scent.sweet': 'Dulce',
        'scent.spicy': 'Especiado',
        'scent.herbal': 'Herbal',
        'scent.fruity': 'Afrutado',
        'scent.green': 'Verde',
        'scent.marine': 'Marino',
        'scent.musky': 'Almizclado',
        'scent.amber': 'Ámbar',
        'intensity.low': 'Bajo',
        'intensity.medium': 'Medio',
        'intensity.high': 'Alto'
    },
    'fr': {
        'model.loading': 'Chargement du modèle...',
        'model.ready': 'Modèle prêt',
        'model.error': 'Erreur du modèle',
        'prediction.processing': 'Traitement de la prédiction...',
        'prediction.complete': 'Prédiction terminée',
        'prediction.failed': 'Prédiction échouée',
        'error.invalid_smiles': 'Format SMILES invalide',
        'error.timeout': 'Délai d\'attente dépassé',
        'error.validation_failed': 'Échec de validation d\'entrée',
        'error.processing_failed': 'Échec du traitement',
        'scent.floral': 'Floral',
        'scent.citrus': 'Agrume',
        'scent.woody': 'Boisé',
        'scent.fresh': 'Frais',
        'scent.sweet': 'Sucré',
        'scent.spicy': 'Épicé',
        'scent.herbal': 'Herbal',
        'scent.fruity': 'Fruité',
        'scent.green': 'Vert',
        'scent.marine': 'Marin',
        'scent.musky': 'Musqué',
        'scent.amber': 'Ambre',
        'intensity.low': 'Faible',
        'intensity.medium': 'Moyen',
        'intensity.high': 'Élevé'
    },
    'de': {
        'model.loading': 'Modell wird geladen...',
        'model.ready': 'Modell bereit',
        'model.error': 'Modell-Fehler',
        'prediction.processing': 'Vorhersage wird verarbeitet...',
        'prediction.complete': 'Vorhersage abgeschlossen',
        'prediction.failed': 'Vorhersage fehlgeschlagen',
        'error.invalid_smiles': 'Ungültiges SMILES-Format',
        'error.timeout': 'Zeitüberschreitung',
        'error.validation_failed': 'Eingabevalidierung fehlgeschlagen',
        'error.processing_failed': 'Verarbeitung fehlgeschlagen',
        'scent.floral': 'Blumig',
        'scent.citrus': 'Zitrus',
        'scent.woody': 'Holzig',
        'scent.fresh': 'Frisch',
        'scent.sweet': 'Süß',
        'scent.spicy': 'Würzig',
        'scent.herbal': 'Kräuterartig',
        'scent.fruity': 'Fruchtig',
        'scent.green': 'Grün',
        'scent.marine': 'Marin',
        'scent.musky': 'Moschusartig',
        'scent.amber': 'Amber',
        'intensity.low': 'Niedrig',
        'intensity.medium': 'Mittel',
        'intensity.high': 'Hoch'
    },
    'ja': {
        'model.loading': 'モデルを読み込み中...',
        'model.ready': 'モデル準備完了',
        'model.error': 'モデルエラー',
        'prediction.processing': '予測を処理中...',
        'prediction.complete': '予測完了',
        'prediction.failed': '予測失敗',
        'error.invalid_smiles': '無効なSMILES形式',
        'error.timeout': 'タイムアウト',
        'error.validation_failed': '入力検証失敗',
        'error.processing_failed': '処理失敗',
        'scent.floral': 'フローラル',
        'scent.citrus': 'シトラス',
        'scent.woody': 'ウッディ',
        'scent.fresh': 'フレッシュ',
        'scent.sweet': 'スウィート',
        'scent.spicy': 'スパイシー',
        'scent.herbal': 'ハーバル',
        'scent.fruity': 'フルーティ',
        'scent.green': 'グリーン',
        'scent.marine': 'マリン',
        'scent.musky': 'ムスキー',
        'scent.amber': 'アンバー',
        'intensity.low': '弱',
        'intensity.medium': '中',
        'intensity.high': '強'
    },
    'zh': {
        'model.loading': '正在加载模型...',
        'model.ready': '模型已就绪',
        'model.error': '模型错误',
        'prediction.processing': '正在处理预测...',
        'prediction.complete': '预测完成',
        'prediction.failed': '预测失败',
        'error.invalid_smiles': '无效的SMILES格式',
        'error.timeout': '请求超时',
        'error.validation_failed': '输入验证失败',
        'error.processing_failed': '处理失败',
        'scent.floral': '花香',
        'scent.citrus': '柑橘',
        'scent.woody': '木香',
        'scent.fresh': '清新',
        'scent.sweet': '甜香',
        'scent.spicy': '辛香',
        'scent.herbal': '草本',
        'scent.fruity': '果香',
        'scent.green': '绿叶',
        'scent.marine': '海洋',
        'scent.musky': '麝香',
        'scent.amber': '琥珀',
        'intensity.low': '低',
        'intensity.medium': '中',
        'intensity.high': '高'
    }
}


class I18nManager:
    """Robust internationalization manager with fallbacks."""
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = DEFAULT_TRANSLATIONS.copy()
        self.fallback_chain = ['en']  # Always fall back to English
        
        # Load external translation files if available
        self._load_external_translations()
    
    def _load_external_translations(self):
        """Load external translation files."""
        try:
            # Look for translation files in locale directory
            locale_dir = Path(__file__).parent.parent / 'locale'
            
            if locale_dir.exists():
                for lang_file in locale_dir.glob('*.json'):
                    lang_code = lang_file.stem
                    try:
                        with open(lang_file, 'r', encoding='utf-8') as f:
                            external_translations = json.load(f)
                            
                        # Merge with defaults
                        if lang_code in self.translations:
                            self.translations[lang_code].update(external_translations)
                        else:
                            self.translations[lang_code] = external_translations
                            
                        logging.debug(f"Loaded translations for {lang_code}")
                        
                    except Exception as e:
                        logging.warning(f"Failed to load translations for {lang_code}: {e}")
                        
        except Exception as e:
            logging.debug(f"No external translations loaded: {e}")
    
    def set_language(self, language: str):
        """Set current language."""
        lang_code = self._normalize_language(language)
        
        if lang_code in self.translations:
            self.current_language = lang_code
            logging.debug(f"Language set to {lang_code}")
        else:
            logging.warning(f"Language {lang_code} not available, using {self.current_language}")
    
    def _normalize_language(self, language: str) -> str:
        """Normalize language code."""
        if not language:
            return self.default_language
        
        # Convert to lowercase and take first 2 characters
        lang_code = language.lower()[:2]
        
        # Map common variations
        language_aliases = {
            'en': 'en',
            'english': 'en',
            'es': 'es',
            'spanish': 'es',
            'español': 'es',
            'fr': 'fr',
            'french': 'fr',
            'français': 'fr',
            'de': 'de',
            'german': 'de',
            'deutsch': 'de',
            'ja': 'ja',
            'japanese': 'ja',
            '日本語': 'ja',
            'zh': 'zh',
            'chinese': 'zh',
            '中文': 'zh'
        }
        
        return language_aliases.get(lang_code, lang_code)
    
    @lru_cache(maxsize=1000)
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Translate a key with robust fallback and caching."""
        target_lang = language or self.current_language
        target_lang = self._normalize_language(target_lang)
        
        # Try target language first
        translation = self._get_translation(key, target_lang)
        
        # Fall back through fallback chain
        if translation == key:
            for fallback_lang in self.fallback_chain:
                if fallback_lang != target_lang:
                    translation = self._get_translation(key, fallback_lang)
                    if translation != key:
                        break
        
        # Format with provided kwargs if no caching conflict
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError, TypeError) as e:
                logging.warning(f"Translation formatting failed for '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, language: str) -> str:
        """Get translation for specific language."""
        if language in self.translations:
            return self.translations[language].get(key, key)
        return key
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        return list(self.translations.keys())
    
    def has_translation(self, key: str, language: Optional[str] = None) -> bool:
        """Check if translation exists for key."""
        target_lang = language or self.current_language
        target_lang = self._normalize_language(target_lang)
        
        return (target_lang in self.translations and 
                key in self.translations[target_lang])
    
    def add_translations(self, language: str, translations: Dict[str, str]):
        """Add translations for a language."""
        lang_code = self._normalize_language(language)
        
        if lang_code in self.translations:
            self.translations[lang_code].update(translations)
        else:
            self.translations[lang_code] = translations.copy()
        
        # Clear cache when translations are updated
        self.translate.cache_clear()
        
        logging.debug(f"Added {len(translations)} translations for {lang_code}")
    
    def format_scent_descriptors(self, descriptors: List[str], 
                               language: Optional[str] = None) -> List[str]:
        """Format scent descriptors with translation."""
        if not descriptors:
            return []
        
        formatted = []
        
        for descriptor in descriptors:
            if not descriptor:
                continue
                
            key = f"scent.{descriptor.lower()}"
            translated = self.translate(key, language)
            
            # If no translation found, use original with proper capitalization
            if translated == key:
                formatted.append(descriptor.capitalize())
            else:
                formatted.append(translated)
        
        return formatted
    
    def get_chemical_family_name(self, family: str, 
                                language: Optional[str] = None) -> str:
        """Get localized chemical family name."""
        if not family:
            return self.translate('chemical_family.unknown', language)
        
        key = f"chemical_family.{family.lower()}"
        translated = self.translate(key, language)
        
        # If no translation found, return properly formatted original
        if translated == key:
            return family.replace('_', ' ').title()
        
        return translated
    
    def get_intensity_label(self, intensity: float, 
                          language: Optional[str] = None) -> str:
        """Get intensity label based on numeric value."""
        if intensity < 3:
            return self.translate('intensity.low', language)
        elif intensity < 7:
            return self.translate('intensity.medium', language)
        else:
            return self.translate('intensity.high', language)
    
    def get_status_message(self, status: str, 
                          language: Optional[str] = None) -> str:
        """Get localized status message."""
        key = f"status.{status.lower()}"
        return self.translate(key, language)
    
    def export_translations(self, language: str, output_path: Path):
        """Export translations for a language to file."""
        lang_code = self._normalize_language(language)
        
        if lang_code not in self.translations:
            raise ValueError(f"No translations available for {lang_code}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.translations[lang_code], f, indent=2, ensure_ascii=False)
            
            logging.info(f"Exported {lang_code} translations to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to export translations: {e}")
            raise


# Global I18n manager instance
i18n_manager = I18nManager()

# Convenience functions for backward compatibility
def get_available_languages() -> List[str]:
    """Get list of available languages."""
    return i18n_manager.get_available_languages()

def translate_text(key: str, language: str = 'en', **kwargs) -> str:
    """Translate a text key to the specified language."""
    return i18n_manager.translate(key, language, **kwargs)

def format_scent_descriptors(descriptors: List[str], language: str = 'en') -> List[str]:
    """Format scent descriptors in the specified language."""
    return i18n_manager.format_scent_descriptors(descriptors, language)

def get_chemical_family_name(family: str, language: str = 'en') -> str:
    """Get localized chemical family name."""
    return i18n_manager.get_chemical_family_name(family, language)