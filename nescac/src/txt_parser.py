import logging
import re
from typing import List, Dict, Optional, Tuple
from base_parser import BaseParser


class TXTParser(BaseParser):
    """Parser for TXT format swimming meet results with unified results list."""
    def __init__(self):
        super().__init__()
        # Improved pattern for prelims entries (rank, name, year, school, seed, prelim)
        # Name: allow for multiple spaces, initials, etc. Year: FR, SR, 04, etc. School: flexible
        self.txt_prelims_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z0-9]{2}|[A-Za-z]{2,4}|\d{2}|)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)(?:\s+([\d:.NTXb#&A-Z!]+))?'
        )
        # New pattern for 2013+ format prelims entries (rank, name, year, school, seed, prelim)
        self.txt_prelims_seed_prelim_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z0-9]{2}|[A-Za-z]{2,4}|\d{2}|)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)'
        )
        # Improved pattern for finals entries (rank, name, year, school, prelim, final, points)
        self.txt_finals_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z0-9]{2}|[A-Za-z]{2,4}|\d{2}|)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?'
        )
        # Improved pattern for single time entries (rank, name, year, school, time)
        self.txt_single_time_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z0-9]{2}|[A-Za-z]{2,4}|\d{2}|)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)'
        )
        # Fallback patterns for lines with NO year field
        self.txt_prelims_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)(?:\s+([\d:.NTXb#&A-Z!]+))?\s*$'
        )
        # New pattern for 2013+ format prelims entries without year (rank, name, school, seed, prelim)
        self.txt_prelims_seed_prelim_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*$'
        )
        self.txt_finals_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)(?:\s+(\d+))?\s*$'
        )
        self.txt_single_time_no_year_re = re.compile(
            r'^\s*(\d+)\s+([A-Za-z.,\'\- ]+?)\s{2,}([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s*$'
        )

    def preprocess_text(self, text: str) -> str:
        """Remove noise lines and strip trailing whitespace."""
        lines = text.split("\n")
        skip_patterns = [
            r"^Licensed to",
            r"^HY-TEK'S MEET MANAGER",
            r"^\d{4}.*Championships.*Results$",
            r"^={40,}$",
            r"^\s*Page\s+\d+",
            r"^\s*www\.",
            r"^\s*NESCAC:\s*\*",
            r"^\s*Pool:\s*[#@]",
            r"^\s*Meet:\s*&",
            r"^\s*\d+\.\d+\s+NAT[AB]$",
            r"^\s*Name\s+Year\s+School",
            r"^\s*Name\s+Year\s+School\s+Prelims\s+Finals\s+Points",
        ]
        processed = []
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                continue
            processed.append(line)
        return "\n".join(processed)

    def _is_section_header(self, line: str) -> Optional[str]:
        """Identify prelims/finals section headers."""
        lc = line.strip().lower()
        exact = {
            'championship final': 'finals',
            'consolation final': 'finals',
            'preconsolation final': 'finals',
            'preliminaries': 'prelims',
            'final': 'finals',
            'finals': 'finals',
            'prelim': 'prelims',
            'prelims': 'prelims'
        }
        if lc in exact:
            return exact[lc]
        if 'preliminaries' in lc:
            return 'prelims'
        if any(ind in lc for ind in ['championship final', 'consolation final', 'preconsolation final']):
            return 'finals'
        if re.match(r'^[a-z] - final$', lc) or re.match(r'^[a-z] - consolation$', lc):
            return 'finals'
        return None

    def _clean_time_string(self, time_str: str) -> str:
        """Remove NATA/NATB indicators and # from time strings."""
        if not time_str:
            return time_str
        # Remove NATA/NATB indicators and #
        cleaned = re.sub(r'\s*NAT[AB]\s*$', '', time_str.strip())
        cleaned = cleaned.replace('#', '')
        return cleaned.strip()
    
    def _cleanup_school_time(self, entry: Dict) -> Dict:
        """Extract stray time from school field if present and use it as the actual time."""
        school = entry.get('school') or ''
        m = re.search(r"\s+([\d]+:[\d]+\.[\d]+|[\d]+\.[\d]+)\s*$", school)
        if m:
            extracted_time = m.group(1)
            entry['school'] = school[:m.start()].strip()
            
            # Use the extracted time as the appropriate time field
            if entry.get('finals_time') and not entry.get('prelim_time'):
                # If we have finals_time but no prelim_time, the extracted time is likely prelim
                entry['prelim_time'] = extracted_time
            elif entry.get('prelim_time') and not entry.get('finals_time'):
                # If we have prelim_time but no finals_time, the extracted time is likely finals
                entry['finals_time'] = extracted_time
            elif not entry.get('prelim_time') and not entry.get('finals_time'):
                # If we have no times at all, use the extracted time as prelim
                entry['prelim_time'] = extracted_time
            else:
                # If we have both times, the extracted time might be a better finals time
                # Check if the extracted time is faster (better) than current finals time
                try:
                    extracted_float = float(extracted_time.replace(':', ''))
                    current_finals = entry.get('finals_time', '')
                    if current_finals:
                        current_float = float(current_finals.replace(':', ''))
                        if extracted_float < current_float:
                            entry['finals_time'] = extracted_time
                except (ValueError, TypeError):
                    # If we can't compare, just use as prelim if no prelim exists
                    if not entry.get('prelim_time'):
                        entry['prelim_time'] = extracted_time
        return entry

    def _parse_year_field(self, year_str: str) -> str:
        """
        Parse year field robustly. Returns 'NONE' if no valid year found.
        
        Args:
            year_str: The year field from the regex match
            
        Returns:
            String representation of year or 'NONE' if invalid
        """
        if not year_str or not year_str.strip():
            return 'NONE'
        
        year_str = year_str.strip()
        
        # If it's longer than 4 characters, it's likely not a year
        if len(year_str) > 4:
            return 'NONE'
        
        # If it's exactly 2 characters and both are uppercase letters, it's likely a year
        if len(year_str) == 2 and year_str.isalpha() and year_str.isupper():
            return year_str
        
        # If it's exactly 2 characters and both are digits, it's likely NOT a year
        # In swimming results, 2-digit numbers like "04", "05", "06" are typically
        # missing year data, not actual years
        if len(year_str) == 2 and year_str.isdigit():
            return 'NONE'
        
        # If it's 3-4 characters and contains numbers, it's likely not a year
        if len(year_str) >= 3 and any(c.isdigit() for c in year_str):
            return 'NONE'
        
        # If it's 3-4 characters and all uppercase letters, it might be a year
        if len(year_str) >= 3 and year_str.isalpha() and year_str.isupper():
            return year_str
        
        # If it's 1 character and is a digit, it's likely not a year
        if len(year_str) == 1 and year_str.isdigit():
            return 'NONE'
        
        # If it's 1-2 characters and contains numbers but is longer than 2, it's likely not a year
        if len(year_str) > 2 and any(c.isdigit() for c in year_str):
            return 'NONE'
        
        # Default case: assume it's a year if it's 1-4 characters
        return year_str

    def _parse_individual_entry(self, line: str, current_section: str, is_exhibition: bool) -> Optional[Dict]:
        clean = line.strip()
        # Only try finals regexes if section is 'finals'
        if current_section == 'finals':
            m = self.txt_finals_re.match(clean)
            if m:
                groups = m.groups()
                rank, name, yr, school, prelim, final, pts = groups
                parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
                # SKIP: If school is empty and prelim or final is a 2-digit number
                if (not school.strip()) and ((prelim and len(prelim.strip()) == 2 and prelim.strip().isdigit()) or (final and len(final.strip()) == 2 and final.strip().isdigit())):
                    return None
                result = {
                    'name': name.strip(),
                    'yr': parsed_yr,
                    'school': school.strip(),
                    'seed_time': None,
                    'prelim_time': self._clean_time_string(prelim) if prelim else None,
                    'finals_time': self._clean_time_string(final) if final else None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
            m = self.txt_finals_no_year_re.match(clean)
            if m:
                rank, name, school, prelim, final, pts = m.groups()
                # SKIP: If school is empty and prelim or final is a 2-digit number
                if (not school.strip()) and ((prelim and len(prelim.strip()) == 2 and prelim.strip().isdigit()) or (final and len(final.strip()) == 2 and final.strip().isdigit())):
                    return None
                result = {
                    'name': name.strip(),
                    'yr': 'NONE',
                    'school': school.strip(),
                    'seed_time': None,
                    'prelim_time': self._clean_time_string(prelim) if prelim else None,
                    'finals_time': self._clean_time_string(final) if final else None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
        # Only try prelims regexes if section is 'prelims'
        if current_section == 'prelims':
            logging.debug(f"Trying prelims regexes for line: {clean}")
            # Try new 2013+ format first (seed, prelim)
            m = self.txt_prelims_seed_prelim_re.match(clean)
            if m:
                logging.debug(f"2013+ format matched: {clean}")
                groups = m.groups()
                rank, name, yr, school, seed_time, prelim_time = groups
                parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
                # SKIP: If school is empty and seed_time or prelim_time is a 2-digit number
                if (not school.strip()) and ((seed_time and len(seed_time.strip()) == 2 and seed_time.strip().isdigit()) or (prelim_time and len(prelim_time.strip()) == 2 and prelim_time.strip().isdigit())):
                    return None
                result = {
                    'name': name.strip(),
                    'yr': parsed_yr,
                    'school': school.strip(),
                    'seed_time': self._clean_time_string(seed_time),
                    'prelim_time': self._clean_time_string(prelim_time),
                    'finals_time': None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
            m = self.txt_prelims_seed_prelim_no_year_re.match(clean)
            if m:
                rank, name, school, seed_time, prelim_time = m.groups()
                # SKIP: If school is empty and seed_time or prelim_time is a 2-digit number
                if (not school.strip()) and ((seed_time and len(seed_time.strip()) == 2 and seed_time.strip().isdigit()) or (prelim_time and len(prelim_time.strip()) == 2 and prelim_time.strip().isdigit())):
                    return None
                result = {
                    'name': name.strip(),
                    'yr': 'NONE',
                    'school': school.strip(),
                    'seed_time': self._clean_time_string(seed_time),
                    'prelim_time': self._clean_time_string(prelim_time),
                    'finals_time': None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
            # Try legacy format (time1, time2 or single time)
            m = self.txt_prelims_re.match(clean)
            if m:
                groups = m.groups()
                rank, name, yr, school, time1, time2 = groups
                parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
                # SKIP: If school is empty and time1 or time2 is a 2-digit number
                if (not school.strip()) and ((time1 and len(time1.strip()) == 2 and time1.strip().isdigit()) or (time2 and len(time2.strip()) == 2 and time2.strip().isdigit())):
                    return None
                # Fix: if two times, first is seed_time, second is prelim_time; if one, it's prelim_time
                if time2:
                    seed_time = self._clean_time_string(time1)
                    prelim_time = self._clean_time_string(time2)
                else:
                    seed_time = None
                    prelim_time = self._clean_time_string(time1)
                result = {
                    'name': name.strip(),
                    'yr': parsed_yr,
                    'school': school.strip(),
                    'seed_time': seed_time,
                    'prelim_time': prelim_time,
                    'finals_time': None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
            m = self.txt_prelims_no_year_re.match(clean)
            if m:
                rank, name, school, time1, time2 = m.groups()
                # SKIP: If school is empty and time1 or time2 is a 2-digit number
                if (not school.strip()) and ((time1 and len(time1.strip()) == 2 and time1.strip().isdigit()) or (time2 and len(time2.strip()) == 2 and time2.strip().isdigit())):
                    return None
                # Fix: if two times, first is seed_time, second is prelim_time; if one, it's prelim_time
                if time2:
                    seed_time = self._clean_time_string(time1)
                    prelim_time = self._clean_time_string(time2)
                else:
                    seed_time = None
                    prelim_time = self._clean_time_string(time1)
                result = {
                    'name': name.strip(),
                    'yr': 'NONE',
                    'school': school.strip(),
                    'seed_time': seed_time,
                    'prelim_time': prelim_time,
                    'finals_time': None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                result = self._cleanup_school_time(result)
                return result
        # Try single time pattern (with year)
        m = self.txt_single_time_re.match(clean)
        if m:
            groups = m.groups()
            rank, name, yr, school, time = groups
            parsed_yr = self._parse_year_field(yr) if yr else 'NONE'
            # SKIP: If school is empty and time is a 2-digit number
            if (not school.strip()) and (time and len(time.strip()) == 2 and time.strip().isdigit()):
                return None
            if current_section == 'prelims':
                prelim_time = self._clean_time_string(time)
                seed_time = None
                finals_time = None
            else:
                prelim_time = None
                seed_time = None
                finals_time = self._clean_time_string(time)
            result = {
                'name': name.strip(),
                'yr': parsed_yr,
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': finals_time,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        # Try single time pattern (no year)
        m = self.txt_single_time_no_year_re.match(clean)
        if m:
            rank, name, school, time = m.groups()
            # SKIP: If school is empty and time is a 2-digit number
            if (not school.strip()) and (time and len(time.strip()) == 2 and time.strip().isdigit()):
                return None
            if current_section == 'prelims':
                prelim_time = self._clean_time_string(time)
                seed_time = None
                finals_time = None
            else:
                prelim_time = None
                seed_time = None
                finals_time = self._clean_time_string(time)
            result = {
                'name': name.strip(),
                'yr': 'NONE',
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': finals_time,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            result = self._cleanup_school_time(result)
            return result
        return None

    def _parse_entry(self, line: str, current_section: str, event_type: str, **kwargs) -> Optional[Dict]:
        """Parse a single data entry line."""
        if not line.strip():
            return None
        # Skip separators and split-time lines
        if re.match(r'^\s*-{5,}\s*$', line) or re.match(r'^\s*(?:\d+\.\d+\s+){2,}\s*$', line):
            return None
        is_exhibition = line.strip().startswith('--')
        if event_type == 'relay':
            return None  # skip relays entirely
        return self._parse_individual_entry(line, current_section, is_exhibition)

    def parse_meet_text(self, text: str) -> List[Dict]:
        processed = self.preprocess_text(text)
        events: Dict[Tuple[str, str, int, str], Dict] = {}
        current_key = None
        current_section = None

        for line in processed.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('Event'):
                if self._is_any_skipped_event(stripped):
                    current_key = None; current_section = None
                    continue
                key = self._get_event_key(stripped)
                if key:
                    if key != current_key:
                        current_key = key
                        self._ensure_event_exists(events, key, stripped)

                    current_section = None
                    continue
                current_key = None; current_section = None
                continue
            if current_key:
                sec = self._is_section_header(line)
                if sec:
                    current_section = sec
                    continue
                
                # Check for 2013+ format preliminaries section (no explicit header)
                # Look for the header line that contains "Seed" and "Prelims"
                if 'Seed' in line and 'Prelims' in line and 'Name' in line:
                    logging.debug(f"Detected 2013+ prelims section: {line.strip()}")
                    current_section = 'prelims'
                    continue
            if current_section and current_key:
                entry = self._parse_entry(line, current_section, events[current_key]['event_type'])
                if not entry:
                    continue
                # merge into unified map
                result_map = events[current_key].setdefault('results_map', {})
                key_tuple = (entry['name'], entry['yr'], entry['school'])
                existing = result_map.get(key_tuple)
                if not existing:
                    existing = {
                        'name': entry['name'],
                        'yr': entry['yr'],
                        'school': entry['school'],
                        'exhibition': entry['exhibition'],
                        'seed_time': None,
                        'prelim_time': None,
                        'finals_time': None,
                        'prelim_rank': None,
                        'final_rank': None
                    }
                    result_map[key_tuple] = existing
                else:
                    if existing['school'] != entry['school']:
                        logging.warning(f"School mismatch for {entry['name']} ({entry['yr']}): {existing['school']} vs {entry['school']}")
                
                # Update based on section
                if current_section == 'prelims':
                    # For prelims, preserve seed_time if it exists in the entry
                    if entry['seed_time'] is not None:
                        logging.debug(f"Setting seed_time for {entry['name']}: {entry['seed_time']}")
                        existing['seed_time'] = entry['seed_time']
                    existing['prelim_time'] = existing['prelim_time'] or entry['prelim_time']
                    existing['prelim_rank'] = existing['prelim_rank'] or entry['rank']

                else:  # finals
                    # In finals, the first time is prelim, second is final
                    if entry['prelim_time'] and not existing['prelim_time']:
                        existing['prelim_time'] = entry['prelim_time']
                    existing['finals_time'] = entry['finals_time']
                    existing['final_rank'] = entry['rank']


        # After merging by (name, yr, school), do a second pass to merge seed_time by (name, school) if missing
        for event in events.values():
            if event.get('event_type') != 'individual':
                continue
            result_map = event.get('results_map', {})
            # Build a lookup by (name, school) for entries with seed_time
            prelim_seed_lookup = {}
            for k, v in result_map.items():
                if v.get('seed_time'):
                    prelim_seed_lookup[(v['name'], v['school'])] = v['seed_time']
            # Fill in missing seed_time for finals entries
            for k, v in result_map.items():
                if not v.get('seed_time'):
                    fallback = prelim_seed_lookup.get((v['name'], v['school']))
                    if fallback:
                        v['seed_time'] = fallback

            # Debug: Check seed times after merging
            seed_time_count = sum(1 for v in result_map.values() if v.get('seed_time'))
            logging.debug(f"After merging: {seed_time_count}/{len(result_map)} entries have seed times")

            # Ensure all prelims entries are included if they have a seed_time or prelim_time
            # Build a set of (name, school) already in result_map
            existing_keys = set((v['name'], v['school']) for v in result_map.values())
            # Find all prelims entries (by section) in result_map
            prelim_entries = [v for v in result_map.values() if v.get('section') == 'prelims']
            for v in prelim_entries:
                key2 = (v['name'], v['school'])
                if key2 not in existing_keys and (v.get('seed_time') or v.get('prelim_time')):
                    # Add this entry to result_map with a new key (name, NONE, school)
                    result_map[(v['name'], 'NONE', v['school'])] = v
                    existing_keys.add(key2)


        # consolidate and return
        self._consolidate_swimmer_results(events)
        return list(events.values())
