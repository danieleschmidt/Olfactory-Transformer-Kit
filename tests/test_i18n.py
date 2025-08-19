"""Test internationalization and multi-region support."""

import pytest
from unittest.mock import Mock, patch

from olfactory_transformer.utils.i18n import (
    ScentDescriptorTranslator,
    RegionalComplianceManager,
    MultiRegionManager,
    SupportedLanguages
)


class TestScentDescriptorTranslator:
    """Test scent descriptor translation."""
    
    def setup_method(self):
        self.translator = ScentDescriptorTranslator()
        
    def test_translate_descriptor_basic(self):
        """Test basic descriptor translation."""
        # English to Spanish
        assert self.translator.translate_descriptor("fresh", "es") == "fresco"
        assert self.translator.translate_descriptor("sweet", "es") == "dulce"
        
        # English to French
        assert self.translator.translate_descriptor("woody", "fr") == "boisé"
        assert self.translator.translate_descriptor("spicy", "fr") == "épicé"
        
        # English to German
        assert self.translator.translate_descriptor("floral", "de") == "blumig"
        assert self.translator.translate_descriptor("citrus", "de") == "zitrusartig"
        
    def test_translate_descriptor_japanese(self):
        """Test translation to Japanese."""
        assert self.translator.translate_descriptor("rose", "ja") == "バラ"
        assert self.translator.translate_descriptor("vanilla", "ja") == "バニラ"
        
    def test_translate_descriptor_chinese(self):
        """Test translation to Chinese."""
        assert self.translator.translate_descriptor("lavender", "zh") == "薰衣草"
        assert self.translator.translate_descriptor("jasmine", "zh") == "茉莉"
        
    def test_translate_descriptor_unknown_language(self):
        """Test handling of unknown language."""
        # Should return original descriptor
        result = self.translator.translate_descriptor("fresh", "unknown")
        assert result == "fresh"
        
    def test_translate_descriptor_unknown_word(self):
        """Test handling of unknown descriptor."""
        # Should return original word
        result = self.translator.translate_descriptor("nonexistent", "es")
        assert result == "nonexistent"
        
    def test_translate_descriptors_list(self):
        """Test translation of descriptor list."""
        descriptors = ["fresh", "sweet", "floral"]
        translated = self.translator.translate_descriptors(descriptors, "es")
        
        expected = ["fresco", "dulce", "floral"]
        assert translated == expected
        
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.translator.get_supported_languages()
        
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "ja" in languages
        assert "zh" in languages
        
    def test_detect_language_english(self):
        """Test language detection for English."""
        text = "fresh floral sweet woody"
        detected = self.translator.detect_language(text)
        assert detected == "en"
        
    def test_detect_language_spanish(self):
        """Test language detection for Spanish."""
        text = "fresco dulce amaderado"
        detected = self.translator.detect_language(text)
        assert detected == "es"
        
    def test_detect_language_mixed(self):
        """Test language detection for mixed text."""
        # Should detect dominant language
        text = "fresh fresco"  # Mixed English/Spanish
        detected = self.translator.detect_language(text)
        # Should detect one of them (implementation dependent)
        assert detected in ["en", "es"]
        
    def test_case_insensitive_translation(self):
        """Test case insensitive translation."""
        assert self.translator.translate_descriptor("FRESH", "es") == "fresco"
        assert self.translator.translate_descriptor("Fresh", "es") == "fresco"
        assert self.translator.translate_descriptor("fresh", "es") == "fresco"


class TestRegionalComplianceManager:
    """Test regional compliance management."""
    
    def setup_method(self):
        self.compliance = RegionalComplianceManager()
        
    def test_check_compliance_eu_safe(self):
        """Test EU compliance with safe ingredients."""
        ingredients = ["linalool", "limonene", "vanillin"]
        result = self.compliance.check_compliance(ingredients, "EU")
        
        assert result["compliant"] == True
        assert "linalool" in result["labeling_required"]
        assert "limonene" in result["labeling_required"]
        
    def test_check_compliance_eu_restricted(self):
        """Test EU compliance with restricted ingredients."""
        ingredients = ["atranol", "linalool"]
        result = self.compliance.check_compliance(ingredients, "EU")
        
        assert result["compliant"] == False
        assert any("atranol" in restriction for restriction in result["restrictions"])
        
    def test_check_compliance_us_safe(self):
        """Test US compliance with safe ingredients."""
        ingredients = ["linalool", "vanillin", "benzyl alcohol"]
        result = self.compliance.check_compliance(ingredients, "US")
        
        assert result["compliant"] == True
        assert len(result["restrictions"]) == 0
        
    def test_check_compliance_us_restricted(self):
        """Test US compliance with restricted ingredients."""
        ingredients = ["methyl eugenol", "vanillin"]
        result = self.compliance.check_compliance(ingredients, "US")
        
        assert result["compliant"] == False
        assert any("methyl eugenol" in restriction for restriction in result["restrictions"])
        
    def test_check_compliance_unknown_region(self):
        """Test handling of unknown region."""
        with pytest.raises(ValueError, match="Region .* not supported"):
            self.compliance.check_compliance(["linalool"], "UNKNOWN")
            
    def test_get_ifra_category_limits(self):
        """Test IFRA category limits."""
        # Oakmoss is restricted in categories 1-3
        assert self.compliance.get_ifra_category_limits("oakmoss", 1) == 0
        assert self.compliance.get_ifra_category_limits("oakmoss", 4) == 0.1
        
        # Unknown ingredient should return None
        assert self.compliance.get_ifra_category_limits("unknown", 1) is None
        
    def test_case_insensitive_compliance(self):
        """Test case insensitive compliance checking."""
        ingredients = ["ATRANOL", "Linalool"]
        result = self.compliance.check_compliance(ingredients, "EU")
        
        assert result["compliant"] == False
        assert any("ATRANOL" in restriction for restriction in result["restrictions"])
        assert "Linalool" in result["labeling_required"]


class TestMultiRegionManager:
    """Test multi-region deployment management."""
    
    def setup_method(self):
        self.region_manager = MultiRegionManager()
        
    def test_get_optimal_region_us(self):
        """Test optimal region selection for US."""
        assert self.region_manager.get_optimal_region("US") == "us-east-1"
        assert self.region_manager.get_optimal_region("CA") == "us-east-1"
        assert self.region_manager.get_optimal_region("MX") == "us-east-1"
        
    def test_get_optimal_region_eu(self):
        """Test optimal region selection for EU."""
        assert self.region_manager.get_optimal_region("GB") == "eu-west-1"
        assert self.region_manager.get_optimal_region("FR") == "eu-west-1"
        assert self.region_manager.get_optimal_region("DE") == "eu-central-1"
        assert self.region_manager.get_optimal_region("IT") == "eu-west-1"
        
    def test_get_optimal_region_asia(self):
        """Test optimal region selection for Asia."""
        assert self.region_manager.get_optimal_region("JP") == "ap-northeast-1"
        assert self.region_manager.get_optimal_region("KR") == "ap-northeast-1"
        assert self.region_manager.get_optimal_region("SG") == "ap-southeast-1"
        assert self.region_manager.get_optimal_region("AU") == "ap-southeast-1"
        
    def test_get_optimal_region_unknown(self):
        """Test fallback for unknown location."""
        assert self.region_manager.get_optimal_region("UNKNOWN") == "us-east-1"
        
    def test_get_optimal_region_case_insensitive(self):
        """Test case insensitive location handling."""
        assert self.region_manager.get_optimal_region("us") == "us-east-1"
        assert self.region_manager.get_optimal_region("gb") == "eu-west-1"
        
    def test_get_compliance_requirements_us(self):
        """Test compliance requirements for US regions."""
        requirements = self.region_manager.get_compliance_requirements("us-east-1")
        assert "US" in requirements
        assert "IFRA" in requirements
        
    def test_get_compliance_requirements_eu(self):
        """Test compliance requirements for EU regions."""
        requirements = self.region_manager.get_compliance_requirements("eu-west-1")
        assert "EU" in requirements
        assert "IFRA" in requirements
        
    def test_get_compliance_requirements_unknown(self):
        """Test fallback compliance requirements."""
        requirements = self.region_manager.get_compliance_requirements("unknown-region")
        assert requirements == ["IFRA"]


class TestSupportedLanguages:
    """Test SupportedLanguages enum."""
    
    def test_language_codes(self):
        """Test language code values."""
        assert SupportedLanguages.ENGLISH.value == "en"
        assert SupportedLanguages.SPANISH.value == "es"
        assert SupportedLanguages.FRENCH.value == "fr"
        assert SupportedLanguages.GERMAN.value == "de"
        assert SupportedLanguages.JAPANESE.value == "ja"
        assert SupportedLanguages.CHINESE.value == "zh"
        
    def test_all_languages_covered(self):
        """Test that all enum languages are in translator."""
        translator = ScentDescriptorTranslator()
        supported = translator.get_supported_languages()
        
        for lang in SupportedLanguages:
            assert lang.value in supported


class TestI18nIntegration:
    """Test integration between i18n components."""
    
    def setup_method(self):
        self.translator = ScentDescriptorTranslator()
        self.compliance = RegionalComplianceManager()
        self.region_manager = MultiRegionManager()
        
    def test_region_compliance_integration(self):
        """Test integration between region and compliance managers."""
        # Get optimal region for Germany
        region = self.region_manager.get_optimal_region("DE")
        assert region == "eu-central-1"
        
        # Get compliance requirements for that region
        requirements = self.region_manager.get_compliance_requirements(region)
        assert "EU" in requirements
        
        # Check compliance with EU regulations
        ingredients = ["linalool", "limonene"]
        result = self.compliance.check_compliance(ingredients, "EU")
        assert result["compliant"] == True
        
    def test_translation_compliance_integration(self):
        """Test integration between translation and compliance."""
        # Translate descriptors to German
        descriptors = ["rose", "lavender", "vanilla"]
        german_descriptors = self.translator.translate_descriptors(descriptors, "de")
        
        expected = ["rose", "lavendel", "vanille"]
        assert german_descriptors == expected
        
        # These should be safe ingredients
        ingredients = ["rose extract", "lavender oil", "vanillin"]
        result = self.compliance.check_compliance(ingredients, "EU")
        assert result["compliant"] == True
        
    def test_global_localization_workflow(self):
        """Test complete global localization workflow."""
        # 1. Detect user location and get optimal region
        user_location = "FR"
        region = self.region_manager.get_optimal_region(user_location)
        assert region == "eu-west-1"
        
        # 2. Get compliance requirements
        requirements = self.region_manager.get_compliance_requirements(region)
        assert "EU" in requirements
        
        # 3. Translate descriptors to French
        descriptors = ["fresh", "floral", "woody"]
        french_descriptors = self.translator.translate_descriptors(descriptors, "fr")
        assert french_descriptors == ["frais", "floral", "boisé"]
        
        # 4. Check compliance for the region
        ingredients = ["linalool", "limonene", "vanillin"]
        compliance = self.compliance.check_compliance(ingredients, "EU")
        assert compliance["compliant"] == True
        
        # Complete workflow should work seamlessly
        assert len(french_descriptors) == len(descriptors)
        assert compliance is not None


class TestI18nPerformance:
    """Test i18n performance characteristics."""
    
    def test_translation_performance(self):
        """Test translation performance."""
        import time
        
        translator = ScentDescriptorTranslator()
        descriptors = ["fresh", "sweet", "woody", "floral", "citrus"]
        
        start_time = time.time()
        for _ in range(1000):
            translator.translate_descriptors(descriptors, "es")
        end_time = time.time()
        
        # Should complete 1000 translations quickly
        assert (end_time - start_time) < 1.0
        
    def test_compliance_performance(self):
        """Test compliance checking performance."""
        import time
        
        compliance = RegionalComplianceManager()
        ingredients = ["linalool", "limonene", "vanillin", "benzyl alcohol"]
        
        start_time = time.time()
        for _ in range(1000):
            compliance.check_compliance(ingredients, "EU")
        end_time = time.time()
        
        # Should complete 1000 compliance checks quickly
        assert (end_time - start_time) < 1.0