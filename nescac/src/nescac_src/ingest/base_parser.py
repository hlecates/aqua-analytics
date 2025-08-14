import logging
import re
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseParser(ABC):
    
    def __init__(self):
        # Event header pattern (shared across formats)
        self.event_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+(\d+)\s+Yard\s+([A-Za-z ]+)(?:\s+Time\s+Trial)?$')
        self.diving_re = re.compile(r'^Event\s+(\d+)\s+(Women|Men)\s+([13])\s+mtr\s+Diving')
    
    def is_diving_event(self, event_line: str) -> bool:
        return self.diving_re.match(event_line) is not None
    
    def _cleanup_school_time(self, entry: Dict) -> Dict:
        school = entry.get('school')
        if school:
            time_match = re.search(
                r'\s+([\d]+:[\d]+\.[\d]+|[\d]+\.[\d]+)\s*$',
                school
            )
            if time_match:
                # remove the embedded time from school
                entry['school'] = school[:time_match.start()].strip()
                # always overwrite finals_time
                entry['finals_time'] = time_match.group(1)
        return entry
    
    def _get_event_key(self, event_line: str) -> Optional[Tuple[str, str, int, str]]:
        match = self.event_re.match(event_line)
        if match:
            event_num, gender, distance, stroke = match.groups()
            # Skip Time Trial events and diving
            if ('time trial' in event_line.lower() or 
                self.is_diving_event(event_line)):
                return None
            logging.debug(f">>> Found event: {event_line}")
            return (event_num, gender, int(distance), stroke.strip())
        return None
    
    def _is_any_skipped_event(self, event_line: str) -> bool:
        return (self.is_diving_event(event_line) or 
                'time trial' in event_line.lower() or
                'swim-off' in event_line.lower())
    
    def _ensure_event_exists(self,
                             events_dict: Dict,
                             event_key: Tuple,
                             event_line: str):
        if event_key in events_dict:
            return

        is_relay = any(w in event_line.lower()
                       for w in ('relay', 'medley relay', 'freestyle relay'))

        if is_relay:
            # Relays will just accumulate into a `results` list
            events_dict[event_key] = {
                'event': event_line,
                'results': [],
                'event_type': 'relay'
            }
        else:
            # Individuals get a results_map to merge prelims+finals
            events_dict[event_key] = {
                'event': event_line,
                'results_map': {},
                'event_type': 'individual'
            }

    def _consolidate_swimmer_results(self, events_dict: Dict) -> Dict:
        for key, data in events_dict.items():
            if data.get('event_type') == 'individual':
                # Move the merged map into a list
                merged = list(data.pop('results_map', {}).values())
                data['results'] = merged
        return events_dict

    @abstractmethod
    def parse_meet_text(self, text: str) -> List[Dict]:
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        pass
    
    @abstractmethod
    def _is_section_header(self, line: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def _parse_entry(self, line: str, current_section: str, event_type: str, **kwargs) -> Optional[Dict]:
        pass