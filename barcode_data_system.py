import cv2
from pyzbar.pyzbar import decode
import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

class BarcodeDataSystem:
    def __init__(self, database_path: str = 'sewing_system.db'):
        """Initialize the barcode and data management system"""
        self.conn = sqlite3.connect(database_path)
        self.setup_database()
        
    def setup_database(self):
        """Create necessary database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fabric_parameters (
            barcode_id TEXT PRIMARY KEY,
            fabric_type TEXT,
            width REAL,
            gsm REAL,
            stitch_length REAL,
            created_at TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sewing_sessions (
            session_id TEXT PRIMARY KEY,
            barcode_id TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            total_stitches INTEGER,
            average_alignment REAL,
            defect_count INTEGER,
            FOREIGN KEY (barcode_id) REFERENCES fabric_parameters (barcode_id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TIMESTAMP,
            stitch_count INTEGER,
            alignment_score REAL,
            defect_count INTEGER,
            analysis_data TEXT,
            FOREIGN KEY (session_id) REFERENCES sewing_sessions (session_id)
        )
        ''')
        
        self.conn.commit()

    def create_barcode(self, fabric_data: Dict) -> str:
        """Create a new barcode entry for fabric parameters"""
        barcode_id = f"FB{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO fabric_parameters 
        (barcode_id, fabric_type, width, gsm, stitch_length, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            barcode_id,
            fabric_data['fabric_type'],
            fabric_data['width'],
            fabric_data['gsm'],
            fabric_data['stitch_length'],
            datetime.now()
        ))
        
        self.conn.commit()
        return barcode_id

    def scan_barcode(self, frame: np.ndarray) -> Optional[Dict]:
        """Scan and decode barcode from camera frame"""
        decoded_objects = decode(frame)
        
        for obj in decoded_objects:
            barcode_id = obj.data.decode('utf-8')
            
            # Fetch fabric parameters from database
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT * FROM fabric_parameters WHERE barcode_id = ?
            ''', (barcode_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'barcode_id': result[0],
                    'fabric_type': result[1],
                    'width': result[2],
                    'gsm': result[3],
                    'stitch_length': result[4],
                    'created_at': result[5]
                }
        
        return None

    def start_session(self, barcode_id: str) -> str:
        """Start a new sewing session"""
        session_id = f"SS{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO sewing_sessions 
        (session_id, barcode_id, start_time)
        VALUES (?, ?, ?)
        ''', (session_id, barcode_id, datetime.now()))
        
        self.conn.commit()
        return session_id

    def log_measurement(self, session_id: str, analysis_results: Dict):
        """Log measurement data from frame analysis"""
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO measurements 
        (session_id, timestamp, stitch_count, alignment_score, defect_count, analysis_data)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            datetime.now(),
            analysis_results['stitch_analysis']['count'],
            analysis_results['alignment_analysis']['alignment_score'],
            analysis_results['defect_analysis']['count'],
            json.dumps(analysis_results)
        ))
        
        self.conn.commit()

    def end_session(self, session_id: str):
        """End a sewing session and compute summary statistics"""
        cursor = self.conn.cursor()
        
        # Fetch session measurements
        cursor.execute('''
        SELECT AVG(alignment_score), SUM(stitch_count), SUM(defect_count)
        FROM measurements WHERE session_id = ?
        ''', (session_id,))
        
        avg_alignment, total_stitches, total_defects = cursor.fetchone()
        
        # Update session summary
        cursor.execute('''
        UPDATE sewing_sessions 
        SET end_time = ?,
            total_stitches = ?,
            average_alignment = ?,
            defect_count = ?
        WHERE session_id = ?
        ''', (
            datetime.now(),
            total_stitches or 0,
            avg_alignment or 0,
            total_defects or 0,
            session_id
        ))
        
        self.conn.commit()

    def generate_report(self, session_id: str) -> Dict:
        """Generate a comprehensive report for a session"""
        cursor = self.conn.cursor()
        
        # Fetch session data
        cursor.execute('''
        SELECT s.*, f.*
        FROM sewing_sessions s
        JOIN fabric_parameters f ON s.barcode_id = f.barcode_id
        WHERE s.session_id = ?
        ''', (session_id,))
        
        session_data = cursor.fetchone()
        
        # Fetch measurements
        measurements_df = pd.read_sql_query('''
        SELECT * FROM measurements WHERE session_id = ?
        ''', self.conn, params=(session_id,))
        
        return {
            'session_summary': {
                'session_id': session_data[0],
                'start_time': session_data[2],
                'end_time': session_data[3],
                'total_stitches': session_data[4],
                'average_alignment': session_data[5],
                'defect_count': session_data[6]
            },
            'fabric_parameters': {
                'barcode_id': session_data[7],
                'fabric_type': session_data[8],
                'width': session_data[9],
                'gsm': session_data[10],
                'stitch_length': session_data[11]
            },
            'measurements': measurements_df.to_dict(orient='records'),
            'statistics': {
                'alignment_trend': measurements_df['alignment_score'].tolist(),
                'stitch_count_trend': measurements_df['stitch_count'].tolist(),
                'defect_count_trend': measurements_df['defect_count'].tolist()
            }
        }

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

    def export_data(self, session_id: str, format: str = 'csv') -> str:
        """Export session data to specified format"""
        report = self.generate_report(session_id)
        
        if format == 'csv':
            filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            pd.DataFrame(report['measurements']).to_csv(filename, index=False)
        elif format == 'json':
            filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
                
        return filename

    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get history of recent sessions"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT s.*, f.fabric_type
        FROM sewing_sessions s
        JOIN fabric_parameters f ON s.barcode_id = f.barcode_id
        ORDER BY s.start_time DESC
        LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'barcode_id': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'total_stitches': row[4],
                'average_alignment': row[5],
                'defect_count': row[6],
                'fabric_type': row[7]
            })
            
        return sessions