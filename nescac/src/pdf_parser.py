import logging
import re
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from base_parser import BaseParser


class PDFParser(BaseParser):
    """Parser for PDF format swimming meet results."""
    
    def __init__(self):
        super().__init__()
        # Improved patterns for PDF parsing - more flexible for actual PDF format
        # DQ pattern: rank (---, DQ, DSQ), name (optional), year, school, prelim, finals, points (finals section)
        # Handle both individual DQs (with names) and relay DQs (without names)
        self.pdf_dq_re = re.compile(
            r'^\s*(---|DQ|DSQ|\*?DQ|\*?DSQ)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z]{2,4})\s+([A-Za-z.\'\- ]+?)(?:\s+([\d:.NTXb#&]+))?(?:\s+([\d:.NTXb#&A-Z!]+))?(?:\s+([\d.\s]+))?\s*$'
        )
        # Relay DQ pattern: rank (---), school, year, time, points
        self.pdf_relay_dq_re = re.compile(
            r'^\s*(---|DQ|DSQ)\s+([A-Za-z.\'\- ]+?)\s+([A-Za-z]{1,2})\s+([\d:.NTXb#&]+)\s+([A-Za-z0-9]+)\s*$'
        )
        # Finals pattern: rank, name, year, school, prelim_time, finals_time, points
        # Fixed to handle names with commas like "Tamposi, Jake"
        # Updated to handle asterisk rankings (*1, *2) for ties
        # Updated to handle points format like "25. 50" (with space)
        self.pdf_finals_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z]{2,4})\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s+([\d.\s]+)\s*$'
        )
        # More flexible finals pattern that handles various formats
        self.pdf_finals_flexible_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z]{2,4})\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s+([\d.\s]+)\s*$'
        )
        # Prelims pattern: rank, name, year, school, seed_time, prelim_time, qualifier
        self.pdf_prelims_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z]{2,4})\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*([A-Za-z]*)\s*$'
        )
        # Single time pattern: rank, name, year, school, time
        self.pdf_single_time_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z]{2,4})\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s*$'
        )
        # Fallback patterns for lines without year
        self.pdf_finals_no_year_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s+([\d.\s]+)\s*$'
        )
        self.pdf_prelims_no_year_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s+([\d:.NTXb#&A-Z!]+)\s*([A-Za-z]*)\s*$'
        )
        self.pdf_single_time_no_year_re = re.compile(
            r'^\s*(\*?\d+)\s+([A-Za-z.,\'\- ]+?)\s+([A-Za-z.\'\- ]+?)\s+([\d:.NTXb#&]+)\s*$'
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
            r"^\s*Name\s+Yr\s+School",
            r"^\s*Name\s+Yr\s+School\s+Seed\s+Time\s+Prelim\s+Time",
            r"^\s*Team\s+Relay",
            r"^\d+\)\s+",  # Relay splits
            # Removed: r"^---\s+",     # Disqualified swimmers - now handled by DQ regex
            r"^Consolation\s+Final",
            r"^PreConsolation\s+Final",
            r"^Early\s+take-off",
            r"^Declared\s+false\s+start",
            r"^\d+\.\s+\w+\s+College\s+\d+",  # Team standings
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
            'prelims': 'prelims',
            'championship': 'finals',
            'consolation': 'finals',
            'bonus': 'finals'
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

    def _parse_year_field(self, year_str: str) -> str:
        """Parse year field robustly. Returns 'NONE' if no valid year found."""
        if not year_str or not year_str.strip():
            return 'NONE'
        year_str = year_str.strip()
        # Only accept standard year codes
        if year_str in ['FR', 'SO', 'JR', 'SR']:
            return year_str
        return 'NONE'

    def _merge_name_and_year(self, name: str, yr: str) -> Tuple[str, str]:
        """If yr is not a valid year, merge it into the name."""
        if yr not in ['FR', 'SO', 'JR', 'SR']:
            merged = f"{name.strip()} {yr.strip()}".strip()
            return merged, 'NONE'
        return name.strip(), yr.strip()

    def _parse_individual_entry(self, line: str, current_section: str, is_exhibition: bool) -> Optional[Dict]:
        """Parse a single swimmer entry line."""
        clean = line.strip()
        
        # Skip reaction time lines and other non-swimmer lines
        if re.match(r'^r:[+-]?\d+\.\d+$', clean):
            return None
        if re.match(r'^\d+\)\s+', clean):  # Relay splits
            return None
        # Do NOT skip DQ lines in finals
        if current_section == 'finals':
            m = self.pdf_dq_re.match(clean)
            if m:
                rank, name, yr, school, prelim, final, pts = m.groups()
                # Handle year field as in other patterns
                if yr not in ['FR', 'SO', 'JR', 'SR'] and len(yr) > 2:
                    name = f"{name.strip()} {yr.strip()}"
                    yr = 'NONE'
                    school_parts = school.strip().split(' ', 1)
                    if len(school_parts) >= 2 and school_parts[0] in ['FR', 'SO', 'JR', 'SR']:
                        yr = school_parts[0]
                        school = school_parts[1] if len(school_parts) > 1 else ''
                    else:
                        school = school.strip()
                else:
                    parsed_yr = self._parse_year_field(yr)
                    yr = parsed_yr
                result = {
                    'name': name.strip(),
                    'yr': yr,
                    'school': school.strip(),
                    'seed_time': None,
                    'prelim_time': self._clean_time_string(prelim) if prelim else None,
                    'finals_time': None,  # DQ: no finals time
                    'rank': 'DQ',
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                return result
            
            # Try relay DQ pattern if individual DQ pattern doesn't match
            m = self.pdf_relay_dq_re.match(clean)
            if m:
                rank, school, yr, time, pts = m.groups()
                result = {
                    'name': 'N/A',  # Relay DQs don't have individual names
                    'yr': yr.strip(),
                    'school': school.strip(),
                    'seed_time': None,
                    'prelim_time': None,
                    'finals_time': self._clean_time_string(time) if time else None,
                    'rank': 'DQ',
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                return result
                
        # Try finals patterns first if in finals section
        if current_section == 'finals':
            m = self.pdf_finals_re.match(clean)
            if m:
                rank, name, yr, school, prelim, final, pts = m.groups()
                # Check if the year field is actually part of the name
                # This happens when the regex incorrectly splits "Tamposi, Jake" into name="Tamposi," and yr="Jake"
                if yr not in ['FR', 'SO', 'JR', 'SR'] and len(yr) > 2:
                    # Merge the name and year back together
                    name = f"{name.strip()} {yr.strip()}"
                    yr = 'NONE'
                    # Now we need to re-parse the school field
                    # The school field now contains the actual year and school
                    school_parts = school.strip().split(' ', 1)
                    if len(school_parts) >= 2 and school_parts[0] in ['FR', 'SO', 'JR', 'SR']:
                        yr = school_parts[0]
                        school = school_parts[1] if len(school_parts) > 1 else ''
                    else:
                        school = school.strip()
                else:
                    parsed_yr = self._parse_year_field(yr)
                    yr = parsed_yr
                
                result = {
                    'name': name.strip(),
                    'yr': yr,
                    'school': school.strip(),
                    'seed_time': None,
                    'prelim_time': self._clean_time_string(prelim) if prelim else None,
                    'finals_time': self._clean_time_string(final) if final else None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                return result
                
            m = self.pdf_finals_no_year_re.match(clean)
            if m:
                rank, name, school, prelim, final, pts = m.groups()
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
                return result
                
        # Try prelims patterns if in prelims section
        if current_section == 'prelims':
            m = self.pdf_prelims_re.match(clean)
            if m:
                rank, name, yr, school, seed_time, prelim_time, qualifier = m.groups()
                # Check if the year field is actually part of the name
                if yr not in ['FR', 'SO', 'JR', 'SR'] and len(yr) > 2:
                    # Merge the name and year back together
                    name = f"{name.strip()} {yr.strip()}"
                    yr = 'NONE'
                    # Now we need to re-parse the school field
                    school_parts = school.strip().split(' ', 1)
                    if len(school_parts) >= 2 and school_parts[0] in ['FR', 'SO', 'JR', 'SR']:
                        yr = school_parts[0]
                        school = school_parts[1] if len(school_parts) > 1 else ''
                    else:
                        school = school.strip()
                else:
                    parsed_yr = self._parse_year_field(yr)
                    yr = parsed_yr
                
                result = {
                    'name': name.strip(),
                    'yr': yr,
                    'school': school.strip(),
                    'seed_time': self._clean_time_string(seed_time),
                    'prelim_time': self._clean_time_string(prelim_time),
                    'finals_time': None,
                    'rank': 'exhibition' if is_exhibition else rank,
                    'exhibition': is_exhibition,
                    'section': current_section
                }
                return result
                
            m = self.pdf_prelims_no_year_re.match(clean)
            if m:
                rank, name, school, seed_time, prelim_time, qualifier = m.groups()
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
                return result
                
        # Try single time patterns
        m = self.pdf_single_time_re.match(clean)
        if m:
            rank, name, yr, school, time = m.groups()
            # Check if the year field is actually part of the name
            if yr not in ['FR', 'SO', 'JR', 'SR'] and len(yr) > 2:
                # Merge the name and year back together
                name = f"{name.strip()} {yr.strip()}"
                yr = 'NONE'
                # Now we need to re-parse the school field
                school_parts = school.strip().split(' ', 1)
                if len(school_parts) >= 2 and school_parts[0] in ['FR', 'SO', 'JR', 'SR']:
                    yr = school_parts[0]
                    school = school_parts[1] if len(school_parts) > 1 else ''
                else:
                    school = school.strip()
            else:
                parsed_yr = self._parse_year_field(yr)
                yr = parsed_yr
            
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
                'yr': yr,
                'school': school.strip(),
                'seed_time': seed_time,
                'prelim_time': prelim_time,
                'finals_time': finals_time,
                'rank': 'exhibition' if is_exhibition else rank,
                'exhibition': is_exhibition,
                'section': current_section
            }
            return result
            
        m = self.pdf_single_time_no_year_re.match(clean)
        if m:
            rank, name, school, time = m.groups()
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
        """Parse meet text using two-pass approach like txt parser."""
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
                
                # Check for PDF preliminaries section headers
                if 'Seed' in line and 'Prelim' in line and 'Name' in line:
                    logging.debug(f"Detected PDF prelims section: {line.strip()}")
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

        # Second pass: merge seed times and ensure all prelims entries are included
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

            # Third pass: try to match by just name for missing finals/prelims times
            # Build lookups for missing data
            prelim_lookup = {}  # (name, school) -> prelim data
            finals_lookup = {}  # (name, school) -> finals data
            
            for k, v in result_map.items():
                if v.get('prelim_time') or v.get('seed_time'):
                    prelim_lookup[(v['name'], v['school'])] = {
                        'prelim_time': v.get('prelim_time'),
                        'seed_time': v.get('seed_time'),
                        'prelim_rank': v.get('prelim_rank')
                    }
                if v.get('finals_time'):
                    finals_lookup[(v['name'], v['school'])] = {
                        'finals_time': v.get('finals_time'),
                        'final_rank': v.get('final_rank')
                    }
            
            # Fill in missing data using name-only matching as fallback
            for k, v in result_map.items():
                # If missing prelim data, try to find by name
                if not v.get('prelim_time') and not v.get('seed_time'):
                    for other_name, other_school in prelim_lookup.keys():
                        if v['name'] == other_name:
                            prelim_data = prelim_lookup[(other_name, other_school)]
                            if not v.get('prelim_time') and prelim_data.get('prelim_time'):
                                v['prelim_time'] = prelim_data['prelim_time']
                            if not v.get('seed_time') and prelim_data.get('seed_time'):
                                v['seed_time'] = prelim_data['seed_time']
                            if not v.get('prelim_rank') and prelim_data.get('prelim_rank'):
                                v['prelim_rank'] = prelim_data['prelim_rank']
                            break
                
                # If missing finals data, try to find by name
                if not v.get('finals_time'):
                    for other_name, other_school in finals_lookup.keys():
                        if v['name'] == other_name:
                            finals_data = finals_lookup[(other_name, other_school)]
                            if finals_data.get('finals_time'):
                                v['finals_time'] = finals_data['finals_time']
                            if not v.get('final_rank') and finals_data.get('final_rank'):
                                v['final_rank'] = finals_data['final_rank']
                            break

            # Ensure all prelims entries are included if they have a seed_time or prelim_time
            existing_keys = set((v['name'], v['school']) for v in result_map.values())
            prelim_entries = [v for v in result_map.values() if v.get('section') == 'prelims']
            for v in prelim_entries:
                key2 = (v['name'], v['school'])
                if key2 not in existing_keys and (v.get('seed_time') or v.get('prelim_time')):
                    result_map[(v['name'], 'NONE', v['school'])] = v
                    existing_keys.add(key2)

        # consolidate and return
        self._consolidate_swimmer_results(events)
        return list(events.values())

    def parse_single_pdf(self, pdf_path: Path) -> Optional[List[Dict]]:
        """Parse a single PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
                
                if not all_text.strip():
                    logging.warning(f"No text extracted from {pdf_path}")
                    return None
                    
                return self.parse_meet_text(all_text)
        except Exception as e:
            logging.error(f"Error parsing {pdf_path}: {e}")
            return None
