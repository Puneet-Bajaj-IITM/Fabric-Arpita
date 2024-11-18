from sewing_system_controller import SewingSystemController
import cv2

def main():
    # Initialize the system
    controller = SewingSystemController()
    
    try:
        # Calibrate the system
        if not controller.calibrate_system():
            print("Calibration failed!")
            return
            
        # Create a test fabric entry
        test_fabric = {
            'fabric_type': 'Cotton',
            'width': 150.0,
            'gsm': 200,
            'stitch_length': 2.5
        }
        barcode_id = controller.data_system.create_barcode(test_fabric)
        
        # Start a session
        session_id = controller.start_session(barcode_id)
        print(f"Started session: {session_id}")
        
        # Set reference line for alignment
        controller.set_reference_line((100, 100, 1820, 100))
        
        # Main processing loop
        try:
            while True:
                # Get latest results
                results = controller.get_latest_results()
                if results:
                    # Display frame with visualizations
                    cv2.imshow('Monitoring', results['frame'])
                    
                    # Print analysis results
                    print(f"Alignment Score: {results['analysis']['alignment_analysis']['alignment_score']:.1f}%")
                    print(f"Stitch Count: {results['analysis']['stitch_analysis']['count']}")
                    print(f"Defects: {results['analysis']['defect_analysis']['count']}")
                    
                # Check for exit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
            
        # Stop session and generate report
        report = controller.stop_session()
        
        # Export session data
        csv_file = controller.data_system.export_data(session_id, 'csv')
        print(f"Session data exported to: {csv_file}")
        
        # Print session summary
        print("\nSession Summary:")
        print(f"Total Stitches: {report['session_summary']['total_stitches']}")
        print(f"Average Alignment: {report['session_summary']['average_alignment']:.1f}%")
        print(f"Total Defects: {report['session_summary']['defect_count']}")
        
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
