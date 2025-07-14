import logging
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from pdf_parser import PDFParser
from txt_parser import TXTParser
from data_formatter import DataFormatter

import utils


class MeetDataPipeline:
    def __init__(self, output_base: Path):
        self.output_base = output_base
        self.raw_pdf_dir = output_base / "raw" / "pdfs"
        self.raw_txt_dir = output_base / "raw" / "txts"
        self.clean_dir = output_base / "processed" / "clean"

        # Create directories
        self.raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        self.raw_txt_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging and initialize parsers
        utils.setup_logging()
        self.pdf_parser = PDFParser()
        self.txt_parser = TXTParser()
        self.data_formatter = DataFormatter()

        # Store individual meet DataFrames in memory
        self.individual_meet_dfs = {}

    def parse_single_pdf(self, pdf_path: Path) -> List[Dict]:
        logging.info(f"Parsing PDF: {pdf_path.name}")
        try:
            all_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"

            events = self.pdf_parser.parse_meet_text(all_text)
            return self._process_events(events, pdf_path)

        except Exception as e:
            logging.error(f"Failed to parse {pdf_path.name}: {e}")
            return []

    def parse_single_txt(self, txt_path: Path) -> List[Dict]:
        logging.info(f"Parsing TXT: {txt_path.name}")
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as file:
                all_text = file.read()

            events = self.txt_parser.parse_meet_text(all_text)
            return self._process_events(events, txt_path)

        except Exception as e:
            logging.error(f"Failed to parse {txt_path.name}: {e}")
            return []

    def _process_events(self, events: List[Dict], source_path: Path) -> List[Dict]:
        """Normalize event dicts into flat records with unified results."""
        meet_name = source_path.stem.replace('-complete-results', '').replace('-', ' ').title()
        source_file = source_path.name
        meet_category = source_path.parent.name

        processed_events = []
        for event in events:
            # Extract header info
            match = re.match(
                r'Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+(.+)',
                event.get('event', '')
            )
            if not match:
                continue
            event_num, gender, distance, stroke = match.groups()

            results = event.get('results', [])
            total_results = len(results)
            finals_results = sum(1 for r in results if r.get('finals_time'))
            
            # Debug information
            logging.debug(f"Event {event_num} {gender} {distance}Y {stroke.strip()}: "
                        f"{finals_results}/{total_results} results have finals times")

            base = {
                'event': event_num,
                'meet': meet_name,
                'stroke': stroke.strip(),
                'gender': gender,
                'distance': int(distance),
                'source_file': source_file,
                'meet_category': meet_category,
                'event_type': event.get('event_type', 'individual'),
                'results': results
            }

            processed_events.append(base)

        return processed_events

    def parse_all_files(
        self,
        pdf_paths: Optional[List[Path]] = None,
        txt_paths: Optional[List[Path]] = None,
        save_individual: bool = True,
        parse_pdfs: bool = True,
        parse_txts: bool = True
    ) -> pd.DataFrame:
        # Initialize paths based on parse flags
        if parse_pdfs:
            if pdf_paths is None:
                pdf_paths = list(self.raw_pdf_dir.rglob("*.pdf"))
            else:
                pdf_paths = pdf_paths or []
        else:
            pdf_paths = []
            
        if parse_txts:
            if txt_paths is None:
                txt_paths = list(self.raw_txt_dir.rglob("*.txt"))
            else:
                txt_paths = txt_paths or []
        else:
            txt_paths = []

        total = len(pdf_paths) + len(txt_paths)
        logging.debug(f"Parsing {len(pdf_paths)} PDF and {len(txt_paths)} TXT files ({total} total)")

        all_events = []
        success = 0
        individual_files = []

        if parse_pdfs:
            for p in pdf_paths:
                ev = self.parse_single_pdf(p)
                if ev:
                    all_events.extend(ev)
                    if save_individual:
                        individual_files.append((p.stem, ev))
                    success += 1
        if parse_txts:
            for t in txt_paths:
                ev = self.parse_single_txt(t)
                if ev:
                    all_events.extend(ev)
                    if save_individual:
                        individual_files.append((t.stem, ev))
                    success += 1

        logging.debug(f"Successfully parsed {success}/{total} files, extracted {len(all_events)} events")

        if save_individual and individual_files:
            self._save_individual_files(individual_files)

        if not all_events:
            logging.warning("No events were parsed!")
            return pd.DataFrame()
        return pd.DataFrame(all_events)

    def _save_individual_files(self, individual_files: List[Tuple[str, List[Dict]]]):
        """Save each meet's events to its own CSV file."""
        for meet_name, events in individual_files:
            if not events:
                continue
                
            df = pd.DataFrame(events)
            if df.empty:
                continue
                
            # Create a clean filename
            clean_name = meet_name.replace('_', '-').replace(' ', '-')
            output_path = self.clean_dir / "individual" / f"{clean_name}_parsed.csv"
            
            self.save_data(df, output_path)
            logging.info(f"Saved individual meet file: {output_path.name}")

    def save_data(self, df: pd.DataFrame, output_path: Path) -> Optional[Path]:
        if df.empty:
            logging.warning(f"No data to save for {output_path}")
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.debug(f"Saved data to {output_path}")
        return output_path



    def process_individual_files(self) -> Optional[Path]:
        """Process all individual CSV files and create a combined dataset."""
        logging.info("Processing individual CSV files")
        
        individual_dir = self.clean_dir / "individual"
        if not individual_dir.exists():
            logging.error(f"Individual directory not found at {individual_dir}")
            return None
        
        # Get all CSV files
        csv_files = list(individual_dir.glob("*.csv"))
        if not csv_files:
            logging.error("No CSV files found in individual directory")
            return None
        
        csv_files.sort()  # Sort to process in chronological order
        logging.info(f"Found {len(csv_files)} CSV files to process")
        
        all_results = []
        
        for csv_file in csv_files:
            logging.info(f"Processing {csv_file.name}...")
            
            try:
                # Load the CSV file
                df = pd.read_csv(csv_file)
                logging.info(f"  Loaded {len(df)} rows from {csv_file.name}")
                
                # Process the dataframe using the new formatter
                clean_df = self.data_formatter.clean_dataframe(df)
                logging.info(f"  Processed into {len(clean_df)} clean rows")
                
                # Add the original results column back to the cleaned dataframe
                clean_df['results'] = df['results']
                
                # Add to results
                all_results.append(clean_df)
                
            except Exception as e:
                logging.error(f"  Error processing {csv_file.name}: {e}")
                continue
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            logging.info(f"Combined dataset has {len(combined_df)} total rows")
            
            # Sort by year, event_name, gender
            combined_df = combined_df.sort_values(['year', 'event_name', 'gender'])
            
            # Save the combined dataset
            output_path = self.clean_dir / "combined_individual_events.csv"
            self.save_data(combined_df, output_path)
            
            # Print summary statistics
            logging.info(f"Dataset Summary:")
            logging.info(f"  Total events: {len(combined_df)}")
            logging.info(f"  Years: {combined_df['year'].min()} - {combined_df['year'].max()}")
            logging.info(f"  Unique events: {combined_df['event_name'].nunique()}")
            logging.info(f"  Genders: {combined_df['gender'].unique()}")
            
            return output_path
        else:
            logging.error("No data was successfully processed")
            return None

    def analyze_cutoffs(self, df: pd.DataFrame) -> None:
        """Analyze the cutoff data to check for any issues."""
        logging.info("Analyzing cutoff data...")
        
        # Check for missing cutoffs
        missing_a = df['a_final_cutoff_sec'].isna().sum()
        missing_b = df['b_final_cutoff_sec'].isna().sum()
        missing_c = df['c_final_cutoff_sec'].isna().sum()
        
        logging.info(f"  Missing A Final cutoffs: {missing_a}")
        logging.info(f"  Missing B Final cutoffs: {missing_b}")
        logging.info(f"  Missing C Final cutoffs: {missing_c}")
        
        # Check events with fewer than 24 swimmers
        small_events = df[df['total_swimmers'] < 24]
        logging.info(f"  Events with <24 swimmers: {len(small_events)}")
        
        if len(small_events) > 0:
            logging.info(f"  Sample small events:")
            for _, row in small_events.head(5).iterrows():
                logging.info(f"    {row['year']} {row['event_name']} {row['gender']}: {row['total_swimmers']} swimmers")
        
        # Check for any cutoff time anomalies
        logging.info(f"Cutoff Time Ranges:")
        for cutoff_type in ['a_final_cutoff_sec', 'b_final_cutoff_sec', 'c_final_cutoff_sec']:
            valid_times = df[df[cutoff_type].notna()][cutoff_type]
            if len(valid_times) > 0:
                logging.info(f"  {cutoff_type}: {valid_times.min():.2f}s - {valid_times.max():.2f}s")

    def run_pipeline(self, parse_pdfs: bool = True, parse_txts: bool = True) -> Tuple[Optional[Path], Optional[Path]]:
        logging.info("Starting meet data pipeline")
        # Parse all files and save individual meet files
        df = self.parse_all_files(parse_pdfs=parse_pdfs, parse_txts=parse_txts, save_individual=True)
        parsed_path = self.save_data(df, self.clean_dir / "parsed_events.csv")
        
        # Process individual files and create combined dataset
        combined_path = self.process_individual_files()
        
        return parsed_path, combined_path


def main():
    base = Path(__file__).parent.parent
    pipeline = MeetDataPipeline(base / "data")

    import sys
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Parse command line arguments
    parse_pdfs = True
    parse_txts = True
    
    if "--pdf" in args:
        parse_pdfs = True
        parse_txts = False
    elif "--txt" in args:
        parse_pdfs = False
        parse_txts = True
    
    # Remove the file type flags from args for command processing
    args = [arg for arg in args if arg not in ["--pdf", "--txt"]]
    
    cmd = args[0].lower() if args else None
    if cmd == "--parse":
        df = pipeline.parse_all_files(save_individual=True, parse_pdfs=parse_pdfs, parse_txts=parse_txts)
        path = pipeline.save_data(df, pipeline.clean_dir / "parsed_events.csv")
        print(f"Success: {path}" if path else "Failed to save parsed data.")
        print("Individual meet files have been saved to the clean/individual directory.")
    elif cmd == "--clean":
        print("The --clean command has been deprecated. Use --combine to process individual files.")
    elif cmd == "--combine":
        combined_path = pipeline.process_individual_files()
        if combined_path:
            print(f"Success: Generated combined dataset at {combined_path}")
            # Load and analyze the combined dataset
            combined_df = pd.read_csv(combined_path)
            pipeline.analyze_cutoffs(combined_df)
        else:
            print("Failed to generate combined dataset.")
    elif cmd == "--analyze":
        combined_path = pipeline.clean_dir / "combined_individual_events.csv"
        if combined_path.exists():
            combined_df = pd.read_csv(combined_path)
            pipeline.analyze_cutoffs(combined_df)
        else:
            print(f"Combined dataset not found at {combined_path}")
            print("Run --combine first to generate the dataset.")
    else:
        parsed, combined = pipeline.run_pipeline(parse_pdfs=parse_pdfs, parse_txts=parse_txts)
        if parsed and combined:
            print(f"Success: Generated {parsed.name}, {combined.name}")
        elif parsed:
            print(f"Partial success: Generated {parsed.name}")
        else:
            print("Pipeline failed.")

if __name__ == "__main__":
    main()
