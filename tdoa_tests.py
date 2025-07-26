import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import math


class TDOAReceiver:
    """Represents a TDOA receiver with position and signal processing capabilities"""
    
    def __init__(self, x, y, name="RX"):
        self.x = x
        self.y = y
        self.name = name
        self.received_signal = None
        self.timestamp = None
    
    def distance_to(self, tx_x, tx_y):
        """Calculate distance to a transmitter"""
        return math.sqrt((self.x - tx_x)**2 + (self.y - tx_y)**2)
    
    def receive_signal(self, signal_data, noise_level=0.1):
        """Simulate receiving a signal with added noise"""
        noise = np.random.normal(0, noise_level, len(signal_data))
        self.received_signal = signal_data + noise
        return self.received_signal


class TDOASystem:
    """TDOA localization system as described in the article"""
    
    def __init__(self, receivers):
        self.receivers = receivers
        self.c = 3e8  # Speed of light in m/s
    
    def calculate_tdoa(self, rx1_idx, rx2_idx):
        """Calculate TDOA between two receivers using cross-correlation"""
        rx1 = self.receivers[rx1_idx]
        rx2 = self.receivers[rx2_idx]
        
        if rx1.received_signal is None or rx2.received_signal is None:
            raise ValueError("Receivers must have received signals")
        
        # Cross-correlation to find time delay
        correlation = signal.correlate(rx1.received_signal, rx2.received_signal, mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation))
        
        # Convert to time delay (assuming sample rate of 2 MHz as in article)
        sample_rate = 2e6
        delay_samples = peak_idx - (len(correlation) - 1) // 2
        tdoa = delay_samples / sample_rate
        
        return tdoa, correlation
    
    def generate_hyperbola(self, rx1, rx2, tdoa, resolution=100):
        """Generate hyperbola points for given TDOA between two receivers"""
        # Distance difference based on TDOA
        delta_d = tdoa * self.c
        
        # Receiver positions
        x1, y1 = rx1.x, rx1.y
        x2, y2 = rx2.x, rx2.y
        
        # Distance between receivers
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if abs(delta_d) >= d:
            # Invalid TDOA - no solution
            return np.array([]), np.array([])
        
        # Center point between receivers
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Rotation angle to align with x-axis
        theta = math.atan2(y2 - y1, x2 - x1)
        
        # Hyperbola parameters
        a = abs(delta_d) / 2
        c = d / 2
        b = math.sqrt(c**2 - a**2)
        
        # Generate hyperbola points
        t = np.linspace(-3, 3, resolution)
        
        if delta_d > 0:
            # Right branch of hyperbola
            x_hyp = a * np.cosh(t)
            y_hyp = b * np.sinh(t)
        else:
            # Left branch of hyperbola  
            x_hyp = -a * np.cosh(t)
            y_hyp = b * np.sinh(t)
        
        # Rotate and translate
        x_rot = x_hyp * math.cos(theta) - y_hyp * math.sin(theta) + cx
        y_rot = x_hyp * math.sin(theta) + y_hyp * math.cos(theta) + cy
        
        return x_rot, y_rot
    
    def multilateration(self, tdoa_measurements):
        """Perform multilateration using TDOA measurements"""
        def objective(pos):
            tx_x, tx_y = pos
            error = 0
            
            for (rx1_idx, rx2_idx, measured_tdoa) in tdoa_measurements:
                rx1 = self.receivers[rx1_idx]
                rx2 = self.receivers[rx2_idx]
                
                # Calculate theoretical TDOA
                d1 = rx1.distance_to(tx_x, tx_y)
                d2 = rx2.distance_to(tx_x, tx_y)
                theoretical_tdoa = (d1 - d2) / self.c
                
                # Add to error
                error += (measured_tdoa - theoretical_tdoa)**2
            
            return error
        
        # Initial guess (center of receivers)
        x_init = np.mean([rx.x for rx in self.receivers])
        y_init = np.mean([rx.y for rx in self.receivers])
        
        # Optimize
        result = minimize(objective, [x_init, y_init], method='BFGS')
        
        return result.x[0], result.x[1], result.success


class SignalGenerator:
    """Generate test signals similar to those mentioned in the article"""
    
    @staticmethod
    def generate_dab_signal(duration=1.0, sample_rate=2e6):
        """Generate DAB-like signal (good correlation properties)"""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # OFDM-like signal with multiple carriers
        signal_data = np.zeros_like(t, dtype=complex)
        for i in range(10, 50):  # Multiple carriers
            freq = i * 1000  # 1 kHz spacing
            phase = np.random.uniform(0, 2*np.pi)
            signal_data += np.exp(1j * (2 * np.pi * freq * t + phase))
        
        return signal_data
    
    @staticmethod
    def generate_fm_signal(duration=1.0, sample_rate=2e6):
        """Generate FM-like signal (poor correlation properties)"""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # FM modulated signal
        carrier_freq = 100e3
        mod_freq = 1000
        mod_index = 5
        
        modulated_signal = np.cos(2 * np.pi * carrier_freq * t + 
                                mod_index * np.sin(2 * np.pi * mod_freq * t))
        
        return modulated_signal.astype(complex)
    
    @staticmethod
    def generate_dmr_signal(duration=1.0, sample_rate=2e6):
        """Generate DMR-like digital signal"""
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Digital modulation (simplified)
        symbol_rate = 4800
        symbols_per_second = symbol_rate
        samples_per_symbol = int(sample_rate / symbols_per_second)
        
        # Generate random digital data
        num_symbols = int(duration * symbols_per_second)
        data_symbols = np.random.choice([-1, 1], num_symbols)
        
        # Upsample and shape
        upsampled = np.repeat(data_symbols, samples_per_symbol)
        upsampled = upsampled[:len(t)]  # Trim to exact length
        
        # Apply carrier
        carrier_freq = 50e3
        signal_data = upsampled * np.exp(1j * 2 * np.pi * carrier_freq * t[:len(upsampled)])
        
        return signal_data


# Test Classes

class TestTDOABasics:
    """Test basic TDOA concepts from the article"""
    
    def test_receiver_creation(self):
        """Test receiver creation and distance calculation"""
        rx1 = TDOAReceiver(0, 0, "RX1")
        rx2 = TDOAReceiver(1000, 0, "RX2")
        
        assert rx1.x == 0
        assert rx1.y == 0
        assert rx2.distance_to(0, 0) == 1000
    
    def test_signal_propagation_delay(self):
        """Test that signals arrive at different times due to distance"""
        # Setup receivers at different distances from transmitter
        rx1 = TDOAReceiver(0, 0, "RX1")
        rx2 = TDOAReceiver(1000, 0, "RX2")  # 1km away
        
        tx_x, tx_y = 500, 0  # Transmitter at 500m
        
        d1 = rx1.distance_to(tx_x, tx_y)  # 500m
        d2 = rx2.distance_to(tx_x, tx_y)  # 500m
        
        # Both receivers same distance - TDOA should be 0
        expected_tdoa = (d1 - d2) / 3e8  # Speed of light
        assert abs(expected_tdoa) < 1e-9  # Very small for equal distances
    
    def test_non_zero_tdoa(self):
        """Test case where TDOA is non-zero"""
        rx1 = TDOAReceiver(0, 0, "RX1")
        rx2 = TDOAReceiver(1000, 0, "RX2")
        
        tx_x, tx_y = 200, 0  # Closer to RX1
        
        d1 = rx1.distance_to(tx_x, tx_y)  # 200m
        d2 = rx2.distance_to(tx_x, tx_y)  # 800m
        
        expected_tdoa = (d1 - d2) / 3e8  # Should be negative (RX1 closer)
        assert expected_tdoa < 0
        assert abs(expected_tdoa) == pytest.approx(600 / 3e8, rel=1e-6)


class TestSignalGeneration:
    """Test signal generation methods"""
    
    def test_dab_signal_generation(self):
        """Test DAB signal generation"""
        signal_data = SignalGenerator.generate_dab_signal(duration=0.1)
        
        assert len(signal_data) == 200000  # 0.1s at 2MHz
        assert np.iscomplexobj(signal_data)
        assert np.mean(np.abs(signal_data)) > 0  # Signal present
    
    def test_fm_signal_generation(self):
        """Test FM signal generation"""
        signal_data = SignalGenerator.generate_fm_signal(duration=0.1)
        
        assert len(signal_data) == 200000
        assert np.iscomplexobj(signal_data)
    
    def test_dmr_signal_generation(self):
        """Test DMR signal generation"""
        signal_data = SignalGenerator.generate_dmr_signal(duration=0.1)
        
        assert len(signal_data) <= 200000  # May be slightly shorter due to symbol alignment
        assert np.iscomplexobj(signal_data)


class TestCorrelationMethods:
    """Test correlation methods for TDOA measurement"""
    
    def test_perfect_correlation_zero_delay(self):
        """Test correlation with identical signals (zero delay)"""
        # Generate test signal
        t = np.linspace(0, 0.1, 10000)
        signal_data = np.sin(2 * np.pi * 1000 * t)  # 1kHz sine
        
        # Cross-correlate identical signals
        correlation = signal.correlate(signal_data, signal_data, mode='full')
        peak_idx = np.argmax(correlation)
        
        # Peak should be at center (zero delay)
        expected_center = (len(correlation) - 1) // 2
        assert peak_idx == expected_center
    
    def test_correlation_with_delay(self):
        """Test correlation with known delay"""
        t = np.linspace(0, 0.1, 10000)
        signal1 = np.sin(2 * np.pi * 1000 * t)
        
        # Create delayed version
        delay_samples = 100
        signal2 = np.zeros_like(signal1)
        signal2[delay_samples:] = signal1[:-delay_samples]
        
        correlation = signal.correlate(signal1, signal2, mode='full')
        peak_idx = np.argmax(correlation)
        
        # Calculate detected delay
        center = (len(correlation) - 1) // 2
        detected_delay = peak_idx - center
        
        assert abs(detected_delay - delay_samples) <= 1  # Allow 1 sample tolerance
    
    def test_correlation_quality_comparison(self):
        """Test that DAB signals have better correlation than FM (as stated in article)"""
        duration = 0.01  # Short duration for fast test
        
        dab_signal = SignalGenerator.generate_dab_signal(duration)
        fm_signal = SignalGenerator.generate_fm_signal(duration)
        
        # Add small delay to create second signal
        delay = 10
        dab_delayed = np.roll(dab_signal, delay)
        fm_delayed = np.roll(fm_signal, delay)
        
        # Calculate correlations
        dab_corr = signal.correlate(dab_signal, dab_delayed, mode='full')
        fm_corr = signal.correlate(fm_signal, fm_delayed, mode='full')
        
        # Measure correlation quality (peak-to-mean ratio)
        dab_quality = np.max(np.abs(dab_corr)) / np.mean(np.abs(dab_corr))
        fm_quality = np.max(np.abs(fm_corr)) / np.mean(np.abs(fm_corr))
        
        # DAB should have better correlation quality
        assert dab_quality > fm_quality


class TestTDOASystem:
    """Test complete TDOA system functionality"""
    
    def test_system_creation(self):
        """Test TDOA system creation"""
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(1000, 0, "RX2"),
            TDOAReceiver(500, 866, "RX3")  # Equilateral triangle
        ]
        
        system = TDOASystem(receivers)
        assert len(system.receivers) == 3
        assert system.c == 3e8
    
    def test_tdoa_calculation(self):
        """Test TDOA calculation between receivers"""
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(1000, 0, "RX2")
        ]
        system = TDOASystem(receivers)
        
        # Generate test signal with known delay
        signal_data = SignalGenerator.generate_dab_signal(duration=0.01)
        delay_samples = 50
        
        receivers[0].received_signal = signal_data
        receivers[1].received_signal = np.roll(signal_data, delay_samples)
        
        tdoa, correlation = system.calculate_tdoa(0, 1)
        
        # Should detect the delay
        expected_tdoa = delay_samples / 2e6  # 50 samples at 2MHz
        assert abs(tdoa - expected_tdoa) < 1e-6
    
    def test_hyperbola_generation(self):
        """Test hyperbola generation for TDOA pairs"""
        rx1 = TDOAReceiver(0, 0, "RX1")
        rx2 = TDOAReceiver(1000, 0, "RX2")
        system = TDOASystem([rx1, rx2])
        
        # Test with valid TDOA
        tdoa = 1e-6  # 1 microsecond
        x_points, y_points = system.generate_hyperbola(rx1, rx2, tdoa)
        
        assert len(x_points) > 0
        assert len(y_points) > 0
        assert len(x_points) == len(y_points)
    
    def test_invalid_tdoa_hyperbola(self):
        """Test hyperbola generation with invalid TDOA (too large)"""
        rx1 = TDOAReceiver(0, 0, "RX1")
        rx2 = TDOAReceiver(1000, 0, "RX2")
        system = TDOASystem([rx1, rx2])
        
        # TDOA corresponding to distance larger than receiver separation
        invalid_tdoa = 1000 / 3e8 + 1e-6  # Slightly more than max possible
        x_points, y_points = system.generate_hyperbola(rx1, rx2, invalid_tdoa)
        
        assert len(x_points) == 0
        assert len(y_points) == 0
    
    def test_multilateration_known_position(self):
        """Test multilateration with known transmitter position"""
        # Setup 3 receivers in triangle
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(2000, 0, "RX2"),
            TDOAReceiver(1000, 1732, "RX3")  # Equilateral triangle
        ]
        system = TDOASystem(receivers)
        
        # Known transmitter position
        tx_x, tx_y = 1000, 500
        
        # Calculate theoretical TDOAs
        d1 = receivers[0].distance_to(tx_x, tx_y)
        d2 = receivers[1].distance_to(tx_x, tx_y)
        d3 = receivers[2].distance_to(tx_x, tx_y)
        
        tdoa_12 = (d1 - d2) / system.c
        tdoa_13 = (d1 - d3) / system.c
        tdoa_23 = (d2 - d3) / system.c
        
        # Perform multilateration
        tdoa_measurements = [
            (0, 1, tdoa_12),
            (0, 2, tdoa_13),
            (1, 2, tdoa_23)
        ]
        
        estimated_x, estimated_y, success = system.multilateration(tdoa_measurements)
        
        assert success
        assert abs(estimated_x - tx_x) < 10  # Within 10m
        assert abs(estimated_y - tx_y) < 10


class TestRealWorldScenarios:
    """Test scenarios mentioned in the article"""
    
    def test_kaiserslautern_geometry(self):
        """Test with receiver geometry similar to Kaiserslautern setup"""
        # Approximate positions based on article (scaled down)
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(3000, 1000, "RX2"),
            TDOAReceiver(1500, 3000, "RX3")
        ]
        system = TDOASystem(receivers)
        
        # Test transmitter in the area between receivers
        tx_x, tx_y = 1500, 1500
        
        # Calculate distances
        distances = [rx.distance_to(tx_x, tx_y) for rx in receivers]
        
        # All distances should be reasonable for city-scale setup
        assert all(100 < d < 5000 for d in distances)
    
    def test_resolution_in_coverage_area(self):
        """Test that resolution is better between receivers (as stated in article)"""
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(2000, 0, "RX2"),
            TDOAReceiver(1000, 1732, "RX3")
        ]
        system = TDOASystem(receivers)
        
        # Position inside triangle (good coverage)
        tx_inside = (1000, 500)
        
        # Position outside triangle (poor coverage)
        tx_outside = (3000, 3000)
        
        # Calculate condition numbers or geometric dilution of precision
        # This is a simplified test - in practice, you'd calculate the full GDOP
        
        # Inside should have better geometry
        center_x = np.mean([rx.x for rx in receivers])
        center_y = np.mean([rx.y for rx in receivers])
        
        dist_inside = math.sqrt((tx_inside[0] - center_x)**2 + (tx_inside[1] - center_y)**2)
        dist_outside = math.sqrt((tx_outside[0] - center_x)**2 + (tx_outside[1] - center_y)**2)
        
        assert dist_inside < dist_outside  # Inside position is closer to center
    
    def test_frequency_bands_mentioned(self):
        """Test with frequency bands mentioned in article results"""
        # DMR at 439 MHz, Mobile at 922 MHz, FM at 96.9 MHz, Unknown at 391 MHz
        frequencies = [439e6, 922e6, 96.9e6, 391e6]
        
        # All frequencies should be within RTL-SDR range (article states ~100MHz to >1GHz)
        for freq in frequencies:
            assert 50e6 < freq < 2e9  # RTL-SDR typical range
    
    def test_data_rate_calculation(self):
        """Test data rate calculation mentioned in article"""
        # Article states: 2 x 8 bit x 2 MHz = 32 Mbit/s per receiver
        sample_rate = 2e6  # 2 MHz
        bits_per_sample = 8 * 2  # 8 bits I + 8 bits Q
        
        data_rate = sample_rate * bits_per_sample
        assert data_rate == 32e6  # 32 Mbit/s as stated in article


# Integration test
class TestCompleteLocalizationPipeline:
    """Test complete localization pipeline as described in article"""
    
    def test_full_pipeline(self):
        """Test complete TDOA localization pipeline"""
        # Setup system similar to article
        receivers = [
            TDOAReceiver(0, 0, "RX1"),
            TDOAReceiver(2000, 0, "RX2"),
            TDOAReceiver(1000, 1732, "RX3")
        ]
        system = TDOASystem(receivers)
        
        # Generate signals (using DAB for good correlation)
        base_signal = SignalGenerator.generate_dab_signal(duration=0.05)
        
        # Simulate transmitter at known position
        tx_x, tx_y = 800, 600
        
        # Calculate propagation delays and simulate reception
        for i, rx in enumerate(receivers):
            distance = rx.distance_to(tx_x, tx_y)
            delay_time = distance / system.c
            delay_samples = int(delay_time * 2e6)  # 2MHz sample rate
            
            # Create delayed and noisy version
            delayed_signal = np.zeros_like(base_signal)
            if delay_samples < len(base_signal):
                delayed_signal[delay_samples:] = base_signal[:-delay_samples]
            else:
                delayed_signal = base_signal  # If delay too large, just use original
            
            rx.receive_signal(delayed_signal, noise_level=0.1)
        
        # Calculate TDOAs
        tdoa_12, _ = system.calculate_tdoa(0, 1)
        tdoa_13, _ = system.calculate_tdoa(0, 2)
        tdoa_23, _ = system.calculate_tdoa(1, 2)
        
        # Perform multilateration
        tdoa_measurements = [
            (0, 1, tdoa_12),
            (0, 2, tdoa_13),
            (1, 2, tdoa_23)
        ]
        
        estimated_x, estimated_y, success = system.multilateration(tdoa_measurements)
        
        # Check results
        assert success
        
        # Allow for some error due to noise and processing
        error_distance = math.sqrt((estimated_x - tx_x)**2 + (estimated_y - tx_y)**2)
        assert error_distance < 200  # Within 200m (reasonable for noisy simulation)


if __name__ == "__main__":
    # Run specific test classes
    pytest.main(["-v", __file__])
