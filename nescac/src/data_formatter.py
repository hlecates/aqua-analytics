import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Optional, Union, Tuple
import logging

class DataFormatter:
    def __init__(self):
        pass

    def clean_time_string(self, time_str: Union[str, None]) -> str:
        if pd.isna(time_str) or not time_str:
            return ""
        
        time_str = str(time_str).strip()
        
        # Remove special characters that indicate records or other annotations
        # Keep only numbers, colons, and periods
        cleaned = re.sub(r'[^0-9:.]', '', time_str)
        
        return cleaned

    def parse_time_to_seconds(self, time_str: str) -> Optional[float]:
        if pd.isna(time_str) or not time_str:
            return None
        
        # Handle special cases
        time_str = str(time_str).strip()
        if time_str in ['NT', 'NTX', 'X', 'DQ', 'NS', 'SCR', '--', '---']:
            return None
        
        # Clean the time string first
        time_str = self.clean_time_string(time_str)
        if not time_str:
            return None
        
        try:
            # Handle MM:SS.HH format
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
            else:
                # Handle SS.HH format
                return float(time_str)
        except:
            return None

    def seconds_to_time_format(self, seconds: float) -> str:
        if seconds is None:
            return ""
        
        if seconds >= 60:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}:{remaining_seconds:06.3f}".rstrip('0').rstrip('.')
        else:
            return f"{seconds:.3f}".rstrip('0').rstrip('.')

    def extract_year_from_source_file(self, source_file: str) -> Optional[int]:
        if pd.isna(source_file) or not source_file:
            return None
        
        source_file = str(source_file)
        if len(source_file) >= 4:
            try:
                return int(source_file[:4])
            except ValueError:
                return None
        return None

    def find_cutoff_rank(self, entries: List[Dict], target_rank: int) -> Optional[float]:
        if not entries or target_rank <= 0:
            return None
        
        # If we have fewer entries than the target rank, return None
        if len(entries) < target_rank:
            return None
        
        # Check if there's a tie at the target rank
        target_entry = entries[target_rank - 1]
        target_time = target_entry.get('prelim_time_sec')
        
        # Check if there are multiple entries with the same time at this rank
        tied_entries = []
        for entry in entries:
            if entry.get('prelim_time_sec') == target_time:
                tied_entries.append(entry)
        
        # If there are ties, use the time of the tie
        if len(tied_entries) > 1:
            return target_time
        
        # If no tie, check if the target rank exists
        if target_rank <= len(entries):
            return target_time
        
        # If target rank doesn't exist, try to average surrounding ranks
        if target_rank > 1 and target_rank < len(entries):
            lower_time = entries[target_rank - 2].get('prelim_time_sec')
            upper_time = entries[target_rank].get('prelim_time_sec')
            
            if lower_time is not None and upper_time is not None:
                return (lower_time + upper_time) / 2
            elif lower_time is not None:
                return lower_time
        
        return None

    def process_entries_list(self, entries: Union[str, List], time_field: str = 'prelim_time') -> List[Dict]:
        # Handle None or NaN values
        if entries is None:
            return []
        
        # Check for pandas NA/NaN only if it's not a list
        if not isinstance(entries, list):
            try:
                if pd.isna(entries):
                    return []
            except:
                pass
        
        # Parse string representation if needed
        if isinstance(entries, str):
            try:
                entries = ast.literal_eval(entries)
            except:
                return []
        
        if not isinstance(entries, list):
            return []
        
        # Clean each entry and add time_sec fields
        cleaned_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            
            # Clean all time fields
            for field in ['seed_time', 'prelim_time', 'finals_time']:
                if field in entry:
                    time_value = entry.get(field)
                    # Clean the time string
                    cleaned_time = self.clean_time_string(time_value)
                    entry[f'{field}_cleaned'] = cleaned_time
                    # Convert to seconds
                    entry[f'{field}_sec'] = self.parse_time_to_seconds(cleaned_time)
                else:
                    entry[f'{field}_cleaned'] = ""
                    entry[f'{field}_sec'] = None
            
            cleaned_entries.append(entry)
        
        return cleaned_entries

    def sort_entries_by_time(self, entries: List[Dict], time_field: str = 'prelim_time_sec') -> List[Dict]:
        # Add original index to preserve tie order
        for i, entry in enumerate(entries):
            entry['_original_index'] = i
        
        # Sort by the specified time field, then by original index
        def get_sort_key(entry):
            time_value = entry.get(time_field)
            if time_value is None:
                return float('inf')
            return time_value
        
        sorted_entries = sorted(entries, key=lambda x: (get_sort_key(x), x['_original_index']))
        
        # Remove the temporary index
        for entry in sorted_entries:
            del entry['_original_index']
        
        return sorted_entries

    def get_winning_time(self, entries: List[Dict]) -> Tuple[Optional[float], str]:
        if not entries:
            return None, ""
        
        # Find the entry with the best (lowest) finals time
        best_entry = None
        best_time = float('inf')
        
        for entry in entries:
            finals_time = entry.get('finals_time_sec')
            if finals_time is not None and finals_time < best_time:
                best_time = finals_time
                best_entry = entry
        
        if best_entry is None:
            return None, ""
        
        winning_time_sec = best_entry.get('finals_time_sec')
        winning_time_format = best_entry.get('finals_time_cleaned', "")
        
        return winning_time_sec, winning_time_format

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Starting to clean dataframe with {len(df)} rows")

        # Filter to individual events only
        df_individual = df[df['event_type'] == 'individual'].copy()
        logging.info(f"Filtered to {len(df_individual)} individual event rows")

        # Extract year from source_file
        df_individual['year'] = df_individual['source_file'].astype(str).apply(self.extract_year_from_source_file)
        
        # Create event_name from distance and stroke with underscore
        df_individual['event_name'] = df_individual.apply(
            lambda row: f"{row['distance']}_{row['stroke']}", axis=1
        )
        
        results = []
        
        for _, row in df_individual.iterrows():
            # Process the results list
            results_data = row['results']
            if pd.isna(results_data):
                continue
            entries = self.process_entries_list(results_data)
            if not entries:
                continue
            
            # Sort entries by prelim time
            sorted_entries = self.sort_entries_by_time(entries, 'prelim_time_sec')
            
            # Get winning time
            winning_time_sec, winning_time_format = self.get_winning_time(entries)
            
            # Get cutoff times
            a_cutoff_sec = self.find_cutoff_rank(sorted_entries, 8)
            b_cutoff_sec = self.find_cutoff_rank(sorted_entries, 16)
            c_cutoff_sec = self.find_cutoff_rank(sorted_entries, 24)
            
            # Convert cutoffs to standard format
            a_cutoff_format = self.seconds_to_time_format(a_cutoff_sec) if a_cutoff_sec is not None else ""
            b_cutoff_format = self.seconds_to_time_format(b_cutoff_sec) if b_cutoff_sec is not None else ""
            c_cutoff_format = self.seconds_to_time_format(c_cutoff_sec) if c_cutoff_sec is not None else ""
            
            # Create result row
            result = {
                'year': row['year'],
                'event_name': row['event_name'],
                'stroke': row['stroke'],
                'gender': row['gender'],
                'distance': row['distance'],
                'meet': row['meet'],
                'source_file': row['source_file'],
                'winning_time_sec': winning_time_sec,
                'winning_time_format': winning_time_format,
                'a_final_cutoff_sec': a_cutoff_sec,
                'a_final_cutoff_format': a_cutoff_format,
                'b_final_cutoff_sec': b_cutoff_sec,
                'b_final_cutoff_format': b_cutoff_format,
                'c_final_cutoff_sec': c_cutoff_sec,
                'c_final_cutoff_format': c_cutoff_format,
                'total_swimmers': len(entries)
            }
            
            results.append(result)
        
        # Create final dataframe
        clean_df = pd.DataFrame(results)
        
        # Sort by year, event_name, gender
        clean_df = clean_df.sort_values(['year', 'event_name', 'gender'])
        
        logging.info(f"Created dataframe with {len(clean_df)} rows")
        
        return clean_df