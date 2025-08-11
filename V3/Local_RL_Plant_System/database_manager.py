import sqlite3
import datetime
from typing import Dict, Any

class DatabaseManager:
    """Manages the SQLite database for logging plant data."""

    def __init__(self, db_path: str):
        """
        Initializes the DatabaseManager.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_table()

    def connect(self):
        """Establishes a connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            # In a real app, you'd want more robust error handling, maybe logging.
            raise

    def create_table(self):
        """Creates the data log table if it doesn't exist."""
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plant_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    soil_moisture REAL,
                    temperature REAL,
                    humidity REAL,
                    light_level REAL,
                    cv_health_score REAL,
                    action_taken INTEGER,
                    reward REAL
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")

    def log_data(self, sensor_data: Dict[str, Any], cv_health_score: float, action: int, reward: float):
        """
        Logs a new entry into the database.

        Args:
            sensor_data: A dictionary of sensor readings.
            cv_health_score: The calculated health score from the CV module.
            action: The action taken by the RL agent.
            reward: The reward calculated for the action.
        """
        if not self.conn:
            print("Cannot log data, no database connection.")
            return

        query = """
            INSERT INTO plant_log (timestamp, soil_moisture, temperature, humidity, light_level, cv_health_score, action_taken, reward)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (
                datetime.datetime.now(),
                sensor_data.get('soil_moisture'),
                sensor_data.get('temperature'),
                sensor_data.get('humidity'),
                sensor_data.get('light_level'),
                cv_health_score,
                action,
                reward
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging data: {e}")

    def get_last_n_events(self, n: int = 5) -> list:
        """
        Retrieves the last N events from the log.

        Args:
            n: The number of events to retrieve.

        Returns:
            A list of tuples representing the last n events.
        """
        if not self.conn:
            return []
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT timestamp, action_taken, cv_health_score FROM plant_log ORDER BY timestamp DESC LIMIT ?", (n,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching last events: {e}")
            return []

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
