import yaml
import time
import numpy as np
import os
from datetime import datetime

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console

from agent import RLAgent
from cv_analyzer import CVAnalyzer
from database_manager import DatabaseManager
from hardware import HardwareController

# --- TUI Generation ---
def generate_layout() -> Layout:
    """Define the layout for the TUI."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(size=10, name="footer"),
    )
    layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))
    return layout

def generate_table(data: dict) -> Table:
    """Generate a table for the TUI from sensor data."""
    table = Table(title="Live System Status")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Timestamp", data.get("timestamp", "N/A"))
    table.add_row("Soil Moisture", str(data.get("soil_moisture", "N/A")))
    table.add_row("Temperature", f"{data.get('temperature', 'N/A')} °C")
    table.add_row("Humidity", f"{data.get('humidity', 'N/A')} %")
    table.add_row("CV Health Score", f"{data.get('cv_health_score', 'N/A'):.2f}%")
    table.add_row("Last Action", data.get("action_str", "N/A"))
    table.add_row("Calculated Reward", f"{data.get('reward', 'N/A'):.2f}")

    return table

# --- Main Application Logic ---
class PlantCareSystem:
    def __init__(self, config):
        self.config = config
        self.console = Console()

        # Create data/log directories if they don't exist
        os.makedirs(os.path.dirname(config['files']['database_path']), exist_ok=True)
        os.makedirs(os.path.dirname(config['files']['log_file_path']), exist_ok=True)

        # Initialize components
        self.hw = HardwareController(config)
        self.cv = CVAnalyzer(config)
        self.db = DatabaseManager(config['files']['database_path'])
        self.agent = RLAgent(config['files']['model_path'])

        self.action_map = {0: "Do Nothing", 1: "Water", 2: "Lights ON", 3: "Water & Lights ON"}
        self.light_is_on = False # Initial state

    def _calculate_reward(self, state):
        """Calculates a reward for logging based on the current state."""
        soil, temp, hum, light, health = state
        ideal = self.config['environment']
        reward = -abs(soil - ideal['ideal_soil_moisture']) / 100.0 \
                 -abs(temp - ideal['ideal_temperature']) \
                 -abs(hum - ideal['ideal_humidity']) / 10.0 \
                 + health / 10.0
        return reward

    def run(self):
        """Starts the main control loop and TUI."""
        layout = generate_layout()
        layout["header"].update(Panel("Autonomous Plant Care System", style="bold green"))

        try:
            with Live(layout, console=self.console, screen=True, redirect_stderr=False) as live:
                while True:
                    # 1. Read Sensors
                    soil_moisture = self.hw.read_soil_moisture()
                    temperature, humidity = self.hw.get_temperature_humidity()

                    # 2. Analyze Plant Health
                    image = self.cv.capture_image("logs/last_capture.jpg")
                    health_score = self.cv.calculate_health_metric(image)

                    # 3. Construct State for RL Agent
                    current_state = np.array([
                        soil_moisture, 
                        temperature, 
                        humidity, 
                        1 if self.light_is_on else 0, 
                        health_score
                    ], dtype=np.float32)

                    # 4. Get Action from Agent
                    action = self.agent.choose_action(current_state)
                    action_str = self.action_map.get(action, "Unknown")

                    # 5. Execute Action
                    if action == 1 or action == 3: # Water
                        self.hw.pump_on()
                    if action == 2 or action == 3: # Light On
                        self.hw.lights_on()
                        self.light_is_on = True
                    else: # Light Off
                        self.hw.lights_off()
                        self.light_is_on = False

                    # 6. Log Data
                    reward = self._calculate_reward(current_state)
                    sensor_data = {
                        'soil_moisture': soil_moisture,
                        'temperature': temperature,
                        'humidity': humidity,
                        'light_level': 1 if self.light_is_on else 0
                    }
                    self.db.log_data(sensor_data, health_score, action, reward)

                    # 7. Update TUI
                    tui_data = {**sensor_data, **{
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cv_health_score": health_score,
                        "action_str": action_str,
                        "reward": reward
                    }}
                    event_log = self.db.get_last_n_events(5)
                    event_panel = Panel("\n".join([f"{r[0]} - {self.action_map[r[1]]} (Health: {r[2]:.2f})" for r in event_log]), title="Recent Events")
                    
                    layout["body"].update(generate_table(tui_data))
                    layout["side"].update(event_panel)

                    # Wait for the next loop
                    time.sleep(self.config['controller']['loop_interval_seconds'])

        except KeyboardInterrupt:
            self.console.print("Shutting down system.")
        finally:
            self.hw.cleanup()
            self.db.close()

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    system = PlantCareSystem(config)
    system.run()
