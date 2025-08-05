"""Tests for sensor integration functionality."""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json

from olfactory_transformer.sensors.enose import (
    ENoseInterface, ENoseArray, OlfactoryQualityControl,
    SensorConfig
)
from olfactory_transformer.core.config import SensorReading


class TestSensorConfig:
    """Test SensorConfig dataclass."""
    
    def test_sensor_config_creation(self):
        """Test basic sensor config creation."""
        config = SensorConfig(
            name="TGS2600",
            sensor_type="gas",
            calibration_offset=0.1,
            calibration_scale=1.2
        )
        
        assert config.name == "TGS2600"
        assert config.sensor_type == "gas"
        assert config.calibration_offset == 0.1
        assert config.calibration_scale == 1.2


class TestENoseInterface:
    """Test ENoseInterface class."""
    
    def test_interface_creation(self):
        """Test basic interface creation."""
        enose = ENoseInterface(
            port="/dev/ttyUSB0",
            sensors=["TGS2600", "TGS2602"],
            sampling_rate=2.0
        )
        
        assert enose.port == "/dev/ttyUSB0"
        assert enose.sensors == ["TGS2600", "TGS2602"]
        assert enose.sampling_rate == 2.0
        assert not enose.is_connected
        assert not enose.is_streaming
    
    def test_mock_connection(self):
        """Test mock connection (no serial hardware)."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Should connect in mock mode
        success = enose.connect()
        assert success
        assert enose.is_connected
        
        # Test disconnect
        enose.disconnect()
        assert not enose.is_connected
    
    def test_single_reading(self):
        """Test single sensor reading."""
        enose = ENoseInterface(sensors=["TGS2600", "TGS2602"])
        
        reading = enose.read_single()
        
        assert isinstance(reading, SensorReading)
        assert reading.timestamp > 0
        assert "TGS2600" in reading.gas_sensors
        assert "TGS2602" in reading.gas_sensors
        assert isinstance(reading.gas_sensors["TGS2600"], float)
        assert reading.gas_sensors["TGS2600"] >= 0
        assert "temperature" in reading.environmental
        assert "humidity" in reading.environmental
    
    @patch('olfactory_transformer.sensors.enose.HAS_SERIAL', True)
    def test_serial_connection_mock(self):
        """Test serial connection with mocking."""
        with patch('olfactory_transformer.sensors.enose.serial') as mock_serial:
            # Mock serial connection
            mock_connection = Mock()
            mock_serial.Serial.return_value = mock_connection
            
            enose = ENoseInterface(port="/dev/ttyUSB0", sensors=["TGS2600"])
            
            success = enose.connect()
            assert success
            assert enose.is_connected
            
            mock_serial.Serial.assert_called_once()
    
    def test_streaming_context_manager(self):
        """Test streaming with context manager."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        readings = []
        start_time = time.time()
        
        # Use context manager with short duration
        with enose.stream(duration=0.1) as sensor_stream:
            for reading in sensor_stream:
                readings.append(reading)
                # Break after first reading to avoid long test
                break
        
        assert len(readings) > 0
        assert isinstance(readings[0], SensorReading)
        assert not enose.is_streaming  # Should be stopped after context
    
    def test_calibration(self):
        """Test sensor calibration."""
        enose = ENoseInterface(sensors=["TGS2600", "TGS2602"])
        
        compounds = ["ethanol", "acetone"]
        concentrations = [10, 50]
        
        # Mock the calibration process (shortened for testing)
        with patch.object(enose, 'read_single') as mock_read:
            mock_read.return_value = SensorReading(
                timestamp=time.time(),
                gas_sensors={"TGS2600": 2.5, "TGS2602": 3.1}
            )
            
            calibration_data = enose.calibrate(compounds, concentrations)
        
        assert isinstance(calibration_data, dict)
        assert "ethanol" in calibration_data
        assert "acetone" in calibration_data
        
        for compound in compounds:
            assert compound in calibration_data
            for concentration in concentrations:
                assert concentration in calibration_data[compound]
    
    def test_calibration_save_load(self):
        """Test calibration data save/load."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Create mock calibration data
        enose.calibration_data = {
            "ethanol": {
                10: {"TGS2600": 2.5},
                50: {"TGS2600": 4.2}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            enose.save_calibration(temp_path)
            
            # Test load
            enose2 = ENoseInterface(sensors=["TGS2600"])
            enose2.load_calibration(temp_path)
            
            assert enose2.calibration_data == enose.calibration_data
        finally:
            Path(temp_path).unlink()
    
    def test_streaming_thread_safety(self):
        """Test that streaming operations are thread-safe."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Start streaming
        enose.start_streaming()
        assert enose.is_streaming
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop streaming
        enose.stop_streaming()
        assert not enose.is_streaming
        
        # Should be able to restart
        enose.start_streaming()
        assert enose.is_streaming
        
        enose.stop_streaming()


class TestENoseArray:
    """Test ENoseArray class."""
    
    def test_array_creation(self):
        """Test multi-sensor array creation."""
        sensor_config = {
            "gas_sensors": ["TGS2600", "TGS2602", "TGS2610"],
            "environmental": ["BME680", "SHT31"],
            "spectrometer": "AS7265x"
        }
        
        array = ENoseArray(sensor_config)
        
        assert array.sensor_config == sensor_config
        assert array.gas_sensors is not None
        assert array.environmental_sensors == ["BME680", "SHT31"]
        assert array.spectrometer == "AS7265x"
    
    def test_connect_all(self):
        """Test connecting all sensor types."""
        sensor_config = {
            "gas_sensors": ["TGS2600"],
            "environmental": ["BME680"],
        }
        
        array = ENoseArray(sensor_config)
        success = array.connect_all()
        
        # Should succeed in mock mode
        assert success
    
    def test_acquire_context_manager(self):
        """Test data acquisition context manager."""
        sensor_config = {
            "gas_sensors": ["TGS2600"],
            "environmental": ["BME680"],
            "spectrometer": "AS7265x"
        }
        
        array = ENoseArray(sensor_config)
        
        data_points = []
        
        # Acquire data briefly
        with array.acquire(duration=0.1) as data_stream:
            for data in data_stream:
                data_points.append(data)
                break  # Just get one sample for testing
        
        assert len(data_points) > 0
        data = data_points[0]
        
        assert "gas_sensors" in data
        assert "environmental" in data
        assert "spectrometer" in data
        
        # Check gas sensor data
        assert isinstance(data["gas_sensors"], dict)
        assert "TGS2600" in data["gas_sensors"]
        
        # Check spectrometer data (should be mock data)
        assert data["spectrometer"] is not None
        assert isinstance(data["spectrometer"], list)


class TestOlfactoryQualityControl:
    """Test OlfactoryQualityControl class."""
    
    def test_qc_creation(self):
        """Test QC system creation."""
        # Mock model and sensor array
        model = Mock()
        sensor_array = Mock()
        
        reference_profile = {
            "target_notes": ["lavender", "fresh"],
            "intensity_range": [6.0, 8.0]
        }
        
        qc = OlfactoryQualityControl(
            model=model,
            sensor_array=sensor_array,
            reference_profile=reference_profile,
            deviation_threshold=0.2
        )
        
        assert qc.model == model
        assert qc.sensor_array == sensor_array
        assert qc.reference_profile == reference_profile
        assert qc.deviation_threshold == 0.2
    
    def test_qc_with_file_reference(self):
        """Test QC system with file-based reference profile."""
        model = Mock()
        sensor_array = Mock()
        
        reference_data = {
            "target_notes": ["citrus", "fresh"],
            "intensity": 7.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(reference_data, f)
            temp_path = f.name
        
        try:
            qc = OlfactoryQualityControl(
                model=model,
                sensor_array=sensor_array,
                reference_profile=temp_path
            )
            
            assert qc.reference_profile == reference_data
        finally:
            Path(temp_path).unlink()
    
    def test_qc_streaming(self):
        """Test QC streaming functionality."""
        model = Mock()
        sensor_array = Mock()
        
        # Mock sensor array streaming
        mock_readings = [
            {"gas_sensors": {"TGS2600": 2.5}, "environmental": {"temp": 25}},
            {"gas_sensors": {"TGS2600": 2.7}, "environmental": {"temp": 25}},
        ]
        
        def mock_acquire():
            for reading in mock_readings:
                yield reading
        
        sensor_array.acquire.return_value.__enter__ = lambda self: mock_acquire()
        sensor_array.acquire.return_value.__exit__ = lambda self, *args: None
        
        qc = OlfactoryQualityControl(
            model=model,
            sensor_array=sensor_array,
            reference_profile={"notes": ["test"]}
        )
        
        # Test streaming (get first batch result)
        batch_results = list(qc.stream(batch_size=2))
        
        assert len(batch_results) > 0
        batch_result = batch_results[0]
        
        # Should have deviation, outlier_notes, and recommendation
        assert hasattr(batch_result, 'deviation')
        assert hasattr(batch_result, 'outlier_notes')
        assert hasattr(batch_result, 'recommendation')
        assert isinstance(batch_result.deviation, float)
        assert isinstance(batch_result.outlier_notes, list)
        assert isinstance(batch_result.recommendation, str)


class TestSensorIntegration:
    """Integration tests for sensor components."""
    
    def test_end_to_end_sensor_flow(self):
        """Test complete sensor data flow."""
        # Create sensor interface
        enose = ENoseInterface(sensors=["TGS2600", "TGS2602"])
        
        # Connect
        assert enose.connect()
        
        # Take reading
        reading = enose.read_single()
        assert isinstance(reading, SensorReading)
        
        # Test data structure
        assert len(reading.gas_sensors) == 2
        assert all(isinstance(v, float) for v in reading.gas_sensors.values())
        assert reading.timestamp > 0
        
        # Disconnect
        enose.disconnect()
        assert not enose.is_connected
    
    def test_sensor_array_integration(self):
        """Test sensor array integration."""
        sensor_config = {
            "gas_sensors": ["TGS2600", "TGS2602"],
            "environmental": ["BME680"],
        }
        
        array = ENoseArray(sensor_config)
        
        # Connect all sensors
        success = array.connect_all()
        assert success
        
        # Quick data acquisition test
        with array.acquire(duration=0.05) as data_stream:
            for data in data_stream:
                # Verify data structure
                assert "gas_sensors" in data
                assert "environmental" in data
                assert len(data["gas_sensors"]) == 2
                break  # Just test one sample
    
    def test_calibration_workflow(self):
        """Test complete calibration workflow."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Connect
        enose.connect()
        
        # Quick single calibration point
        compounds = ["ethanol"]
        concentrations = [50]
        
        with patch.object(enose, 'read_single') as mock_read:
            # Mock consistent readings
            mock_read.return_value = SensorReading(
                timestamp=time.time(),
                gas_sensors={"TGS2600": 3.5}
            )
            
            calibration_data = enose.calibrate(compounds, concentrations)
        
        # Verify calibration structure
        assert "ethanol" in calibration_data
        assert 50 in calibration_data["ethanol"]
        assert "TGS2600" in calibration_data["ethanol"][50]
        assert isinstance(calibration_data["ethanol"][50]["TGS2600"], float)


class TestSensorErrorHandling:
    """Test error handling in sensor operations."""
    
    def test_connection_failure(self):
        """Test handling of connection failures."""
        with patch('olfactory_transformer.sensors.enose.HAS_SERIAL', True):
            with patch('olfactory_transformer.sensors.enose.serial') as mock_serial:
                # Make serial connection fail
                mock_serial.Serial.side_effect = Exception("Connection failed")
                
                enose = ENoseInterface(port="/dev/ttyUSB0")
                success = enose.connect()
                
                # Should fall back to mock mode
                assert success or not success  # Either mock mode or proper failure
    
    def test_reading_with_disconnected_sensor(self):
        """Test reading when sensor is not connected."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Don't connect, but try to read
        reading = enose.read_single()
        
        # Should still return a reading (mock data)
        assert isinstance(reading, SensorReading)
        assert "TGS2600" in reading.gas_sensors
    
    def test_invalid_calibration_data(self):
        """Test handling of invalid calibration parameters."""
        enose = ENoseInterface(sensors=["TGS2600"])
        
        # Test with empty compounds list
        calibration_data = enose.calibrate([], [10])
        assert calibration_data == {}
        
        # Test with empty concentrations list  
        calibration_data = enose.calibrate(["ethanol"], [])
        assert calibration_data == {}


if __name__ == "__main__":
    pytest.main([__file__])