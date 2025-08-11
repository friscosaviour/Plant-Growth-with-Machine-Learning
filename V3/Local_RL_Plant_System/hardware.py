import time
import random

# Attempt to import Raspberry Pi specific libraries
try:
    import RPi.GPIO as GPIO
    import adafruit_dht
    import board
    import spidev
    IS_RASPBERRY_PI = True
except (ImportError, NotImplementedError):
    IS_RASPBERRY_PI = False

class HardwareController:
    """Manages all hardware interactions with sensors and actuators."""

    def __init__(self, config: dict):
        """
        Initializes the HardwareController.

        Args:
            config: A dictionary containing hardware pin configurations.
        """
        self.config = config['hardware']
        if IS_RASPBERRY_PI:
            self._initialize_gpio()
        else:
            print("WARNING: Raspberry Pi libraries not found. Running in mock mode.")

    def _initialize_gpio(self):
        """Sets up GPIO pins and sensors."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.config['pump_pin'], GPIO.OUT)
        GPIO.setup(self.config['light_pin'], GPIO.OUT)
        GPIO.output(self.config['pump_pin'], GPIO.LOW) # Ensure pump is off
        GPIO.output(self.config['light_pin'], GPIO.LOW)  # Ensure light is off

        # Initialize DHT sensor
        self.dht_device = adafruit_dht.DHT22(getattr(board, f"D{self.config['dht_pin']}"), use_pulseio=False)

        # Initialize SPI for ADC (e.g., MCP3008)
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0) # bus 0, device 0
        self.spi.max_speed_hz = 1350000

    def read_soil_moisture(self) -> int:
        """Reads the value from the soil moisture sensor via an ADC."""
        if not IS_RASPBERRY_PI:
            return random.randint(200, 800) # Mock data

        # Read SPI data from MCP3008 channel
        adc = self.spi.xfer2([1, (8 + self.config['soil_moisture_channel']) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data

    def get_temperature_humidity(self) -> (float, float):
        """Reads temperature and humidity from the DHT sensor."""
        if not IS_RASPBERRY_PI:
            return round(random.uniform(18, 25), 1), round(random.uniform(55, 65), 1) # Mock data

        try:
            temperature_c = self.dht_device.temperature
            humidity = self.dht_device.humidity
            if temperature_c is not None and humidity is not None:
                return temperature_c, humidity
        except RuntimeError as error:
            # Errors happen fairly often with DHT sensors, just retry
            print(f"DHT sensor error: {error.args[0]}")
        return self.get_temperature_humidity() # Simple retry

    def pump_on(self, duration_seconds: int = 5):
        """Turns the water pump on for a specified duration."""
        print(f"Turning pump ON for {duration_seconds} seconds.")
        if IS_RASPBERRY_PI:
            GPIO.output(self.config['pump_pin'], GPIO.HIGH)
            time.sleep(duration_seconds)
            GPIO.output(self.config['pump_pin'], GPIO.LOW)
        else:
            time.sleep(duration_seconds) # Mock action
        print("Pump OFF.")

    def lights_on(self):
        """Turns the LED light strip on."""
        print("Turning lights ON.")
        if IS_RASPBERRY_PI:
            GPIO.output(self.config['light_pin'], GPIO.HIGH)

    def lights_off(self):
        """Turns the LED light strip off."""
        print("Turning lights OFF.
")
        if IS_RASPBERRY_PI:
            GPIO.output(self.config['light_pin'], GPIO.LOW)

    def cleanup(self):
        """Cleans up GPIO resources."""
        print("Cleaning up hardware resources.")
        if IS_RASPBERRY_PI:
            GPIO.cleanup()
            self.spi.close()
            self.dht_device.exit()
