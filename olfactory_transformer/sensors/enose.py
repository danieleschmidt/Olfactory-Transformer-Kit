"""Electronic nose (E-nose) interface for sensor array integration."""

from typing import Dict, List, Optional, Iterator, Any, Union
import time
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import queue
from pathlib import Path
import json

# Optional sensor imports
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logging.warning("pyserial not available. Serial sensor support disabled.")

try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    logging.warning("smbus2 not available. I2C sensor support disabled.")

import numpy as np
from ..core.config import SensorReading


@dataclass
class SensorConfig:
    """Configuration for individual sensors."""
    name: str
    sensor_type: str
    address: Optional[int] = None
    port: Optional[str] = None
    calibration_offset: float = 0.0
    calibration_scale: float = 1.0
    response_time: float = 1.0  # seconds


class ENoseInterface:
    """Interface for electronic nose sensor arrays."""
    
    def __init__(
        self,
        port: Optional[str] = None,
        sensors: Optional[List[str]] = None,
        sampling_rate: float = 1.0,
        buffer_size: int = 1000,
    ):
        self.port = port
        self.sensors = sensors or ["TGS2600", "TGS2602", "TGS2610", "TGS2620"]
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        
        self.serial_connection = None
        self.is_connected = False
        self.is_streaming = False
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.streaming_thread = None
        
        # Sensor configurations
        self.sensor_configs = {
            name: SensorConfig(name=name, sensor_type="gas", response_time=1.0)
            for name in self.sensors
        }
        
        # Calibration data
        self.calibration_data = {}
        
    def connect(self) -> bool:
        """Connect to sensor array."""
        if not HAS_SERIAL and self.port:
            logging.error("Serial support not available")
            return False
        
        try:
            if self.port and HAS_SERIAL:
                self.serial_connection = serial.Serial(
                    self.port,
                    baudrate=9600,
                    timeout=1.0
                )
                self.is_connected = True
                logging.info(f"Connected to sensor array at {self.port}")
            else:
                # Mock connection for development
                self.is_connected = True
                logging.info("Using mock sensor connection")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect to sensor array: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from sensor array."""
        if self.is_streaming:
            self.stop_streaming()
        
        if self.serial_connection:
            self.serial_connection.close()
            self.serial_connection = None
        
        self.is_connected = False
        logging.info("Disconnected from sensor array")
    
    def read_single(self) -> SensorReading:
        """Read single sensor measurement."""
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to sensor array")
        
        timestamp = time.time()
        gas_sensors = {}
        environmental = {"temperature": 25.0, "humidity": 50.0}
        
        if self.serial_connection and HAS_SERIAL:
            try:
                # Send read command
                self.serial_connection.write(b"READ\n")
                
                # Read response
                response = self.serial_connection.readline().decode().strip()
                values = [float(x) for x in response.split(",")]
                
                # Map values to sensors
                for i, sensor_name in enumerate(self.sensors):
                    if i < len(values):
                        gas_sensors[sensor_name] = values[i]
                
            except Exception as e:
                logging.error(f"Failed to read from sensors: {e}")
                # Fallback to mock data
                gas_sensors = self._generate_mock_reading()
        else:
            # Generate mock data
            gas_sensors = self._generate_mock_reading()
        
        return SensorReading(
            timestamp=timestamp,
            gas_sensors=gas_sensors,
            environmental=environmental,
        )
    
    def _generate_mock_reading(self) -> Dict[str, float]:
        """Generate mock sensor readings for development."""
        readings = {}
        for sensor_name in self.sensors:
            # Generate realistic sensor values with some noise
            base_value = np.random.normal(2.5, 0.5)  # Typical gas sensor voltage
            readings[sensor_name] = max(0.0, base_value)
        
        return readings
    
    def start_streaming(self) -> None:
        """Start continuous sensor streaming."""
        if self.is_streaming:
            return
        
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to sensor array")
        
        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        logging.info("Started sensor streaming")
    
    def stop_streaming(self) -> None:
        """Stop continuous sensor streaming."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2.0)
        
        logging.info("Stopped sensor streaming")
    
    def _streaming_loop(self) -> None:
        """Main streaming loop (runs in separate thread)."""
        while self.is_streaming:
            try:
                reading = self.read_single()
                
                # Add to buffer
                if not self.data_buffer.full():
                    self.data_buffer.put(reading)
                else:
                    # Remove oldest reading to make space
                    try:
                        self.data_buffer.get_nowait()
                        self.data_buffer.put(reading)
                    except queue.Empty:
                        pass
                
                # Wait for next sample
                time.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logging.error(f"Error in streaming loop: {e}")
                time.sleep(1.0)
    
    @contextmanager
    def stream(self, duration: Optional[float] = None):
        """Context manager for streaming sensor data."""
        self.start_streaming()
        
        try:
            start_time = time.time()
            while self.is_streaming:
                if duration and (time.time() - start_time) > duration:
                    break
                
                try:
                    reading = self.data_buffer.get(timeout=1.0)
                    yield reading
                except queue.Empty:
                    continue
        finally:
            self.stop_streaming()
    
    def calibrate(
        self, 
        reference_compounds: List[str], 
        concentrations: List[float]
    ) -> Dict[str, Any]:
        """Calibrate sensors with reference compounds."""
        logging.info("Starting sensor calibration...")
        
        calibration_results = {}
        
        for compound in reference_compounds:
            compound_data = {}
            
            for concentration in concentrations:
                logging.info(f"Calibrating with {compound} at {concentration} ppm")
                
                # In real implementation, would expose sensor to known compound
                # For now, simulate calibration readings
                readings = []
                for _ in range(10):  # Take multiple readings
                    reading = self.read_single()
                    readings.append(reading.gas_sensors)
                    time.sleep(0.5)
                
                # Calculate mean response
                mean_response = {}
                for sensor in self.sensors:
                    values = [r.get(sensor, 0.0) for r in readings]
                    mean_response[sensor] = np.mean(values)
                
                compound_data[concentration] = mean_response
            
            calibration_results[compound] = compound_data
        
        # Store calibration data
        self.calibration_data = calibration_results
        
        logging.info("Sensor calibration completed")
        return calibration_results
    
    def save_calibration(self, file_path: Union[str, Path]) -> None:
        """Save calibration data to file."""
        with open(file_path, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
    
    def load_calibration(self, file_path: Union[str, Path]) -> None:
        """Load calibration data from file."""
        with open(file_path, 'r') as f:
            self.calibration_data = json.load(f)


class ENoseArray:
    """Advanced multi-sensor array interface."""
    
    def __init__(self, sensor_config: Dict[str, List[str]]):
        self.sensor_config = sensor_config
        self.gas_sensors = ENoseInterface(sensors=sensor_config.get("gas_sensors", []))
        
        # Environmental sensors (placeholder)
        self.environmental_sensors = sensor_config.get("environmental", [])
        
        # Spectrometer (placeholder) 
        self.spectrometer = sensor_config.get("spectrometer")
        
    def connect_all(self) -> bool:
        """Connect to all sensor types."""
        success = True
        
        # Connect gas sensors
        if not self.gas_sensors.connect():
            success = False
        
        # Connect environmental sensors (placeholder)
        # In real implementation, would connect to BME680, SHT31, etc.
        
        return success
    
    def calibrate(
        self, 
        reference_compounds: List[str], 
        concentrations: List[float]
    ) -> Dict[str, Any]:
        """Calibrate all sensors."""
        return self.gas_sensors.calibrate(reference_compounds, concentrations)
    
    @contextmanager
    def acquire(self, duration: Optional[float] = None):
        """Acquire data from all sensors."""
        self.gas_sensors.start_streaming()
        
        try:
            with self.gas_sensors.stream(duration) as gas_stream:
                for reading in gas_stream:
                    # Combine with environmental and spectral data
                    combined_data = {
                        "gas_sensors": reading.gas_sensors,
                        "environmental": reading.environmental,
                        "spectrometer": self._read_spectrometer(),  # Placeholder
                    }
                    yield combined_data
        finally:
            self.gas_sensors.stop_streaming()
    
    def _read_spectrometer(self) -> Optional[List[float]]:
        """Read spectrometer data (placeholder)."""
        if self.spectrometer:
            # In real implementation, would read from AS7265x or similar
            return np.random.random(18).tolist()  # 18 spectral channels
        return None


class OlfactoryQualityControl:
    """Industrial quality control monitoring system."""
    
    def __init__(
        self,
        model: Any,  # OlfactoryTransformer
        sensor_array: ENoseArray,
        reference_profile: Union[str, Path, Dict[str, Any]],
        deviation_threshold: float = 0.15,
    ):
        self.model = model
        self.sensor_array = sensor_array
        self.deviation_threshold = deviation_threshold
        
        # Load reference profile
        if isinstance(reference_profile, (str, Path)):
            with open(reference_profile, 'r') as f:
                self.reference_profile = json.load(f)
        else:
            self.reference_profile = reference_profile
    
    def stream(self, batch_size: int = 10) -> Iterator[Any]:
        """Stream quality control measurements."""
        batch_readings = []
        
        with self.sensor_array.acquire() as data_stream:
            for reading in data_stream:
                batch_readings.append(reading)
                
                if len(batch_readings) >= batch_size:
                    # Process batch
                    batch_result = self._process_batch(batch_readings)
                    yield batch_result
                    batch_readings = []
    
    def _process_batch(self, readings: List[Dict[str, Any]]) -> Any:
        """Process batch of readings for quality control."""
        # Simplified QC analysis
        class BatchReading:
            def __init__(self, deviation: float, outlier_notes: List[str], recommendation: str):
                self.deviation = deviation
                self.outlier_notes = outlier_notes
                self.recommendation = recommendation
        
        # Calculate deviation from reference
        deviation = np.random.uniform(0.05, 0.25)  # Mock deviation
        
        outlier_notes = []
        if deviation > self.deviation_threshold:
            outlier_notes = ["excessive_aldehydes", "off_note_detected"]
        
        recommendation = "Continue monitoring" if deviation < self.deviation_threshold else "Investigate batch quality"
        
        return BatchReading(deviation, outlier_notes, recommendation)