import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy import ndimage
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import re


class SleipnerProcessor:
    """
    Enhanced processor for Sleipner CO2 storage data
    with velocity-based saturation modeling and temporal predictions
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the processor
        
        Parameters:
        -----------
        data_dir : str, optional
            Base directory for data files. If None, paths must be absolute.
        """
        self.data_dir = data_dir
        self.grid_data = None
        self.grid_dimensions = None
        self.plume_data = {}
        self.well_log_data = {}
        self.well_position_data = {}
        self.binary_grid = None
        self.velocity_maps = {}
        self.velocity_diff_maps = {}
        self.saturation_grid = None
        self.metadata = {}
        self.ml_models = {}
        self.results = {}
    
    def get_full_path(self, file_path):
        """Get full path for a file"""
        if self.data_dir and not os.path.isabs(file_path):
            return os.path.join(self.data_dir, file_path)
        return file_path
    
    def load_plume_boundary(self, file_path, layer_name=None):
        """Load a plume boundary file for a specific layer"""
        full_path = self.get_full_path(file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Plume boundary file {full_path} not found")
            return None
        
        # Extract layer name from file name if not provided
        if layer_name is None:
            basename = os.path.basename(file_path)
            layer_match = re.search(r'L(\d+)', basename)
            if layer_match:
                layer_name = f"L{layer_match.group(1)}"
            else:
                layer_name = f"Layer_{len(self.plume_data) + 1}"
        
        print(f"Loading plume boundary from {file_path} as {layer_name}")
        
        # Read file and extract coordinates
        coordinates = []
        segment_id = 0
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Try to identify data format and skip header
        data_section = content
        if '@' in content:
            # Format with @ separator for data section
            data_section = content.split('@')[-1]
        elif '#' in content and 'POINT' in content:
            # Format with # comments and POINT marker
            for line in content.split('\n'):
                if 'POINT' in line:
                    data_start = content.index(line) + len(line)
                    data_section = content[data_start:]
                    break
        
        # Process each line of data
        for line in data_section.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('!') or line.startswith('#'):
                continue
            
            # Parse line for coordinates and segment ID
            parts = line.split()
            if len(parts) >= 2:  # Need at least X, Y
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    # Check for segment ID in different positions
                    if len(parts) >= 4:
                        segment_id = int(parts[3])
                    elif len(parts) == 3 and parts[2].isdigit():
                        segment_id = int(parts[2])
                    coordinates.append((x, y, segment_id))
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Error parsing line: {line}, {e}")
        
        # Group by segment ID
        segments = {}
        for x, y, sid in coordinates:
            if sid not in segments:
                segments[sid] = []
            segments[sid].append((x, y))
        
        # Convert to numpy arrays
        for sid in segments:
            segments[sid] = np.array(segments[sid])
        
        print(f"  Found {len(segments)} segments with {len(coordinates)} points")
        
        # Store in plume_data
        self.plume_data[layer_name] = segments
        
        return segments
    
    def load_grid_data(self, file_path):
        """Load grid data from file"""
        full_path = self.get_full_path(file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Grid file {full_path} not found")
            # Use default grid dimensions from documentation
            self.grid_dimensions = (64, 118, 9)  # nx, ny, nz
            print(f"  Using default grid dimensions: {self.grid_dimensions}")
            return None
        
        print(f"Loading grid data from {file_path}")
        
        # Try to identify file type
        with open(full_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        
        # Check if file looks like GRDECL format
        is_grdecl = any(re.match(r'^[A-Z]+', line) for line in first_lines)
        
        if is_grdecl:
            # Parse as GRDECL format
            grid_data = {}
            current_section = None
            data_values = []
            
            with open(full_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('--') or line.startswith('ECHO') or line.startswith('NOECHO'):
                        continue
                    
                    # Check for section headers (all caps keywords)
                    if re.match(r'^[A-Z]+', line) and not any(c.islower() for c in line if c.isalpha()):
                        # Save previous section data if exists
                        if current_section and data_values:
                            grid_data[current_section] = np.array(data_values)
                            data_values = []
                        
                        current_section = line.split()[0]  # Take the first word as the keyword
                        continue
                    
                    # Check for end of section marker
                    if line == '/':
                        if current_section and data_values:
                            grid_data[current_section] = np.array(data_values)
                            data_values = []
                            current_section = None
                        continue
                    
                    # Process data line
                    if current_section:
                        # Extract numeric values
                        values = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if values:
                            try:
                                numeric_values = [float(val) for val in values]
                                data_values.extend(numeric_values)
                            except ValueError:
                                pass
            
            # Extract grid dimensions
            if 'SPECGRID' in grid_data:
                nx, ny, nz = grid_data['SPECGRID'][:3].astype(int)
                self.grid_dimensions = (nx, ny, nz)
            elif 'GRIDUNIT' in grid_data and len(grid_data['GRIDUNIT']) >= 3:
                nx, ny, nz = grid_data['GRIDUNIT'][:3].astype(int)
                self.grid_dimensions = (nx, ny, nz)
            else:
                # Default dimensions
                self.grid_dimensions = (64, 118, 9)  # Using 9 layers for plume data
            
            print(f"  Grid dimensions: {self.grid_dimensions}")
            print(f"  Grid sections: {list(grid_data.keys())}")
            
            self.grid_data = grid_data
            return grid_data
        else:
            # Try as simple format with dimensions
            try:
                # Read the first few lines to find dimensions
                dimensions = None
                with open(full_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i > 20:  # Only check first 20 lines
                            break
                        
                        values = re.findall(r'\d+', line)
                        if len(values) >= 3:
                            try:
                                dims = [int(v) for v in values[:3]]
                                if all(d > 0 for d in dims):
                                    dimensions = tuple(dims)
                                    break
                            except ValueError:
                                pass
                
                if dimensions:
                    self.grid_dimensions = dimensions
                    print(f"  Grid dimensions: {self.grid_dimensions}")
                    return {'DIMENSIONS': np.array(dimensions)}
                else:
                    print("  Could not determine grid dimensions, using default")
                    self.grid_dimensions = (64, 118, 9)
                    return {'DIMENSIONS': np.array(self.grid_dimensions)}
            except Exception as e:
                print(f"  Error parsing grid file: {e}")
                self.grid_dimensions = (64, 118, 9)
                return {'DIMENSIONS': np.array(self.grid_dimensions)}
    
    def load_well_data(self, file_path, well_name=None, data_type='log'):
        """Load well data (log or position)"""
        full_path = self.get_full_path(file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Well data file {full_path} not found")
            return None
        
        # Extract well name from file name if not provided
        if well_name is None:
            basename = os.path.basename(file_path)
            well_match = re.search(r'(\d+[\-/]\d+[\-A-Z]*\d*)', basename)
            if well_match:
                well_name = well_match.group(1)
            else:
                well_name = f"Well_{len(self.well_log_data) + 1}"
        
        print(f"Loading well {data_type} data for {well_name} from {file_path}")
        
        try:
            # Try to determine file format
            with open(full_path, 'r') as f:
                first_lines = [f.readline() for _ in range(10)]
            
            # Check if it's LAS format
            is_las = any('~V' in line or '~VERSION' in line for line in first_lines)
            
            if is_las and data_type == 'log':
                # Parse as LAS format
                try:
                    # Try to use lasio if available
                    try:
                        import lasio
                        las = lasio.read(full_path)
                        well_df = las.df()
                        print(f"  Loaded with lasio: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                        
                        if data_type == 'log':
                            self.well_log_data[well_name] = well_df
                        else:
                            self.well_position_data[well_name] = well_df
                            
                        return well_df
                    except ImportError:
                        print("  lasio not available, falling back to manual parsing")
                    
                    # Manual parsing of LAS
                    curve_names = []
                    data_rows = []
                    data_section_started = False
                    
                    with open(full_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            
                            # Skip empty lines
                            if not line:
                                continue
                            
                            # Check for curve information section
                            if '~CURVE' in line or '~C ' in line:
                                continue
                            
                            # Check for data section start
                            if line.startswith('~A'):
                                data_section_started = True
                                # Extract curve names if present
                                if ' ' in line:
                                    curve_names = line.replace('~A', '').strip().split()
                                continue
                            
                            # Collect curve names if in curve section
                            if not data_section_started and line.startswith('.'):
                                parts = line.split()
                                if parts:
                                    curve_name = parts[0].strip('.')
                                    if curve_name and curve_name not in curve_names:
                                        curve_names.append(curve_name)
                            
                            # Process data lines
                            if data_section_started:
                                values = line.split()
                                if len(values) == len(curve_names):
                                    # Convert to numeric values
                                    numeric_values = []
                                    for val in values:
                                        try:
                                            numeric_values.append(float(val))
                                        except ValueError:
                                            # Handle missing values
                                            numeric_values.append(np.nan)
                                    data_rows.append(numeric_values)
                    
                    if curve_names and data_rows:
                        # Create DataFrame
                        well_df = pd.DataFrame(data_rows, columns=curve_names)
                        print(f"  Loaded manually: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                        
                        if data_type == 'log':
                            self.well_log_data[well_name] = well_df
                        else:
                            self.well_position_data[well_name] = well_df
                            
                        return well_df
                    else:
                        print("  Error: No curve names or data rows found")
                        return None
                        
                except Exception as e:
                    print(f"  Error loading LAS file: {e}")
                    return None
            
            # Try as position data format
            if data_type == 'position' or not is_las:
                # Check for position data format
                header_line = None
                data_start = 0
                
                for i, line in enumerate(first_lines):
                    if 'Absolute X' in line or 'X Offset' in line or 'MD' in line:
                        header_line = i
                        data_start = i + 1
                        break
                
                if header_line is not None:
                    # Parse as position data
                    col_names = None
                    data_rows = []
                    
                    with open(full_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Extract column names from header line
                    if header_line < len(lines):
                        header = lines[header_line].strip()
                        col_names = re.findall(r'[A-Za-z_]+[\s_]*[A-Za-z]*', header)
                        if not col_names:
                            # Default position column names
                            col_names = ['Absolute_X', 'Absolute_Y', 'X_Offset', 'Y_Offset', 'TVD', 'MD', 'TVDSS']
                    else:
                        # Default position column names
                        col_names = ['Absolute_X', 'Absolute_Y', 'X_Offset', 'Y_Offset', 'TVD', 'MD', 'TVDSS']
                    
                    # Process data lines
                    for i in range(data_start, len(lines)):
                        line = lines[i].strip()
                        if not line or '-----------' in line:
                            continue
                        
                        # Split by whitespace
                        values = line.split()
                        if len(values) >= len(col_names):
                            data_rows.append(values[:len(col_names)])
                    
                    if data_rows:
                        # Create DataFrame
                        well_df = pd.DataFrame(data_rows, columns=col_names)
                        
                        # Convert to numeric
                        for col in well_df.columns:
                            well_df[col] = pd.to_numeric(well_df[col], errors='coerce')
                        
                        print(f"  Loaded position data: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                        
                        if data_type == 'position':
                            self.well_position_data[well_name] = well_df
                        else:
                            self.well_log_data[well_name] = well_df
                            
                        return well_df
                
            # Try as CSV or fixed-width format
            try:
                # First try as CSV
                well_df = pd.read_csv(full_path)
                print(f"  Loaded as CSV: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                
                if data_type == 'log':
                    self.well_log_data[well_name] = well_df
                else:
                    self.well_position_data[well_name] = well_df
                    
                return well_df
            except Exception:
                # Try with different delimiters
                try:
                    well_df = pd.read_csv(full_path, delim_whitespace=True)
                    print(f"  Loaded with whitespace delimiter: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                    
                    if data_type == 'log':
                        self.well_log_data[well_name] = well_df
                    else:
                        self.well_position_data[well_name] = well_df
                        
                    return well_df
                except Exception:
                    # Try as fixed-width
                    try:
                        well_df = pd.read_fwf(full_path)
                        print(f"  Loaded as fixed-width: {well_df.shape[0]} rows, {well_df.shape[1]} columns")
                        
                        if data_type == 'log':
                            self.well_log_data[well_name] = well_df
                        else:
                            self.well_position_data[well_name] = well_df
                            
                        return well_df
                    except Exception as e:
                        print(f"  Error loading well data file: {e}")
                        return None
        
        except Exception as e:
            print(f"  Error loading well data file: {e}")
            return None
    
    
    def load_velocity_map(self, file_path, map_name, map_type='velocity'):
        """
        Load a velocity map file with improved handling for irregular data
        
        Parameters:
        -----------
        file_path : str
            Path to the velocity map file
        map_name : str
            Name to identify this velocity map
        map_type : str
            Type of map ('velocity', 'pre-injection', 'post-injection')
            
        Returns:
        --------
        numpy.ndarray
            2D array of velocity values
        """
        full_path = self.get_full_path(file_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: Velocity map file {full_path} not found")
            return None
        
        print(f"Loading {map_type} map '{map_name}' from {file_path}")
        
        try:
            # Read the file and parse header
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            # Parse header information
            header_info = {}
            header_lines = []
            data_start = 0
            
            # Try to find header information in first few lines
            for i, line in enumerate(lines[:20]):
                header_lines.append(line.strip())
                
                # Look for grid dimensions
                if i == 0 and re.search(r'[\-\d]+\s+[\-\d]+', line):
                    try:
                        values = line.strip().split()
                        header_info['dimensions'] = [int(values[0]), int(values[1])]
                    except (ValueError, IndexError):
                        pass
                
                # Look for boundary coordinates
                if i == 2 and len(line.strip().split()) >= 4:
                    try:
                        values = line.strip().split()
                        header_info['boundaries'] = [float(val) for val in values[:4]]
                    except (ValueError, IndexError):
                        pass
                
                # Detect start of data section (first line with many float values)
                if len(re.findall(r'[\-\d\.]+', line)) > 5:
                    data_start = i
                    break
            
            # Process the data section, handling irregular rows
            velocity_data = []
            max_values_per_row = 0
            
            # First pass: find maximum row length and collect data
            raw_data_rows = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                
                # Extract all numeric values
                values = re.findall(r'[\-\d\.]+', line)
                if values:
                    float_values = [float(val) for val in values]
                    raw_data_rows.append(float_values)
                    max_values_per_row = max(max_values_per_row, len(float_values))
            
            # Second pass: create regular grid by padding rows
            for row in raw_data_rows:
                if len(row) < max_values_per_row:
                    # Pad row with NaN values
                    padded_row = row + [np.nan] * (max_values_per_row - len(row))
                    velocity_data.append(padded_row)
                else:
                    velocity_data.append(row)
            
            # Convert to numpy array
            velocity_array = np.array(velocity_data)
            
            # Check for NaN values and replace with interpolated values if needed
            if np.isnan(velocity_array).any():
                print(f"  Note: Missing values detected in velocity map, filling with interpolation")
                # Simple interpolation: replace NaNs with column means
                col_means = np.nanmean(velocity_array, axis=0)
                for j in range(velocity_array.shape[1]):
                    mask = np.isnan(velocity_array[:, j])
                    velocity_array[mask, j] = col_means[j]
            
            # Store in velocity_maps dictionary
            self.velocity_maps[map_name] = {
                'data': velocity_array,
                'header': header_info,
                'type': map_type
            }
            
            print(f"  Loaded velocity map: {velocity_array.shape}")
            return velocity_array
                
        except Exception as e:
            print(f"  Error loading velocity map: {e}")
            
            # Try alternate approach for irregular data
            try:
                print("  Attempting alternate loading method...")
                # Try to load as a CSV with flexible delimiters
                import pandas as pd
                
                # Read with pandas, which can handle irregular data better
                df = pd.read_csv(full_path, delim_whitespace=True, header=None, skip_blank_lines=True, 
                            skiprows=data_start if data_start > 0 else None,
                            error_bad_lines=False, warn_bad_lines=True)
                
                # Convert to regular numpy array
                velocity_array = df.values
                
                # Store in velocity_maps dictionary
                self.velocity_maps[map_name] = {
                    'data': velocity_array,
                    'header': header_info,
                    'type': map_type
                }
                
                print(f"  Loaded velocity map with alternate method: {velocity_array.shape}")
                return velocity_array
                
            except Exception as inner_e:
                print(f"  Error with alternate loading method: {inner_e}")
                
                # Last resort: try to load just the values, ignoring structure
                try:
                    print("  Attempting basic numeric extraction...")
                    all_values = []
                    with open(full_path, 'r') as f:
                        for line in f:
                            values = re.findall(r'[\-\d\.]+', line)
                            all_values.extend([float(val) for val in values])
                    
                    # Create 1D array and reshape to approximate 2D
                    values_array = np.array(all_values)
                    size = len(values_array)
                    
                    # Try to determine a reasonable shape
                    width = int(np.sqrt(size))
                    height = size // width
                    
                    # Pad if necessary
                    if width * height < size:
                        values_array = values_array[:width * height]
                    
                    velocity_array = values_array.reshape(height, width)
                    
                    # Store in velocity_maps dictionary
                    self.velocity_maps[map_name] = {
                        'data': velocity_array,
                        'header': {'note': 'Approximate reshaping'},
                        'type': map_type
                    }
                    
                    print(f"  Loaded velocity map with basic extraction: {velocity_array.shape}")
                    return velocity_array
                    
                except Exception as final_e:
                    print(f"  All loading methods failed: {final_e}")
                    return None
                
    def normalize_velocity_differences(self, velocity_diff):
        """
        Normalize velocity difference values by removing outliers and clipping to reasonable range
        
        Parameters:
        -----------
        velocity_diff : numpy.ndarray
            Velocity difference map to normalize
            
        Returns:
        --------
        numpy.ndarray
            Normalized velocity difference map
        """
        import numpy as np
        
        # Make a copy to avoid modifying the original
        normalized_diff = velocity_diff.copy()
        
        # Get valid (non-NaN) values
        valid_values = normalized_diff[~np.isnan(normalized_diff)]
        
        if len(valid_values) == 0:
            print("  Warning: No valid velocity difference values found")
            return normalized_diff
        
        # Calculate statistics for valid values
        mean_diff = np.mean(valid_values)
        std_diff = np.std(valid_values)
        
        # Define reasonable limits for velocity differences
        # Values beyond 5 standard deviations are likely outliers
        lower_limit = max(mean_diff - 5 * std_diff, -500)  # Lower limit of -500 m/s
        upper_limit = min(mean_diff + 5 * std_diff, 200)   # Upper limit of 200 m/s
        
        print(f"  Normalizing velocity differences: mean={mean_diff:.2f}, std={std_diff:.2f}")
        print(f"  Clipping to range [{lower_limit:.2f}, {upper_limit:.2f}] m/s")
        
        # Apply limits to remove extreme outliers
        normalized_diff = np.clip(normalized_diff, lower_limit, upper_limit)
        
        return normalized_diff
    
    def calculate_velocity_difference(self, pre_map_name, post_map_name, diff_name=None):
        """
        Calculate difference between pre and post-injection velocity maps
        
        Parameters:
        -----------
        pre_map_name : str
            Name of pre-injection velocity map
        post_map_name : str
            Name of post-injection velocity map
        diff_name : str, optional
            Name for the difference map
            
        Returns:
        --------
        numpy.ndarray
            2D array of velocity differences
        """
        if pre_map_name not in self.velocity_maps:
            print(f"Error: Pre-injection map '{pre_map_name}' not found")
            return None
        
        if post_map_name not in self.velocity_maps:
            print(f"Error: Post-injection map '{post_map_name}' not found")
            return None
        
        pre_map = self.velocity_maps[pre_map_name]['data']
        post_map = self.velocity_maps[post_map_name]['data']
        
        # Check if maps have same shape
        if pre_map.shape != post_map.shape:
            print(f"Error: Map shapes don't match: {pre_map.shape} vs {post_map.shape}")
            
            # Try to resize to match
            try:
                # Resize post_map to match pre_map
                from scipy.ndimage import zoom
                zoom_factors = (pre_map.shape[0] / post_map.shape[0], 
                               pre_map.shape[1] / post_map.shape[1])
                post_map_resized = zoom(post_map, zoom_factors)
                print(f"  Resized post-injection map to {post_map_resized.shape}")
                
                # Calculate difference
                diff_map = post_map_resized - pre_map
            except Exception as e:
                print(f"  Error resizing maps: {e}")
                return None
        else:
            # Calculate difference
            diff_map = post_map - pre_map
        
        # Generate difference map name if not provided
        if diff_name is None:
            diff_name = f"{pre_map_name}_to_{post_map_name}_diff"
        
        # Store difference map
        self.velocity_diff_maps[diff_name] = {
            'data': diff_map,
            'pre_map': pre_map_name,
            'post_map': post_map_name
        }
        
        print(f"Created velocity difference map '{diff_name}': {diff_map.shape}")
        print(f"  Min diff: {np.min(diff_map):.2f}, Max diff: {np.max(diff_map):.2f}")
        
        return diff_map


    def create_binary_plume_grid(self, grid_dims=None):
        """Create a binary grid representing CO2 plume presence"""
        if not self.plume_data:
            print("Warning: No plume data loaded. Load plume boundaries first.")
            return None
        
        # Use provided grid dimensions or loaded dimensions
        if grid_dims is None:
            if self.grid_dimensions is None:
                print("Warning: Grid dimensions not set. Using default (64, 118, 9).")
                nx, ny, nz = 64, 118, 9
            else:
                nx, ny, nz = self.grid_dimensions
                # Limit nz to match number of plume layers
                layer_nums = []
                for layer in self.plume_data:
                    match = re.search(r'L(\d+)', layer)
                    if match:
                        layer_nums.append(int(match.group(1)))
                if layer_nums:
                    nz = max(layer_nums)
        else:
            nx, ny, nz = grid_dims
        
        print(f"Creating binary plume grid with dimensions {nx} x {ny} x {nz}")
        
        # Create empty grid
        binary_grid = np.zeros((nx, ny, nz), dtype=int)
        
        # Collect all coordinates to determine bounds
        all_coords = []
        for layer, segments in self.plume_data.items():
            for seg_id, coords in segments.items():
                all_coords.append(coords)
        
        if not all_coords:
            print("Warning: No plume coordinates found.")
            self.binary_grid = binary_grid
            return binary_grid
        
        # Find coordinate bounds
        all_coords = np.vstack(all_coords)
        x_min, y_min = np.min(all_coords, axis=0)
        x_max, y_max = np.max(all_coords, axis=0)
        
        # Process each layer
        for layer, segments in self.plume_data.items():
            # Extract layer index
            layer_match = re.search(r'L(\d+)', layer)
            if not layer_match:
                print(f"Warning: Could not extract layer number from {layer}")
                continue
            
            layer_idx = int(layer_match.group(1)) - 1
            if layer_idx >= nz:
                print(f"Warning: Layer index {layer_idx} exceeds grid dimensions")
                continue
            
            # Process each segment
            for seg_id, coords in segments.items():
                # Skip empty segments
                if len(coords) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Convert to grid indices
                x_indices = np.round((coords[:, 0] - x_min) / (x_max - x_min) * (nx - 1)).astype(int)
                y_indices = np.round((coords[:, 1] - y_min) / (y_max - y_min) * (ny - 1)).astype(int)
                
                # Ensure valid indices
                x_indices = np.clip(x_indices, 0, nx - 1)
                y_indices = np.clip(y_indices, 0, ny - 1)
                
                # Create polygon
                polygon = np.column_stack((x_indices, y_indices))
                
                # Create convex hull if needed
                try:
                    hull = ConvexHull(polygon)
                    vertices = hull.vertices
                    polygon = polygon[vertices]
                except Exception as e:
                    print(f"  Warning: ConvexHull error for {layer} segment {seg_id}: {e}")
                    if len(polygon) >= 3:
                        # Use the original polygon points if ConvexHull fails
                        pass
                    else:
                        # Skip if not enough points
                        continue
                
                # Create grid coordinates
                x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
                points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
                
                # Check which points are inside the polygon
                path = Path(polygon)
                mask = path.contains_points(points)
                mask = mask.reshape((ny, nx)).T
                
                # Mark cells inside the outline
                binary_grid[:, :, layer_idx][mask] = 1
        
        print(f"Binary grid created: {np.sum(binary_grid)} cells marked as CO2 presence")
        self.binary_grid = binary_grid
        return binary_grid
    


    def convert_velocity_to_saturation(self, diff_map_name=None, method='linear', params=None):
        """
        Convert velocity difference to CO2 saturation with improved normalization
        
        Parameters:
        -----------
        diff_map_name : str, optional
            Name of velocity difference map to use
        method : str
            Conversion method ('linear', 'gassmann', 'custom')
        params : dict, optional
            Parameters for the conversion method
            
        Returns:
        --------
        numpy.ndarray
            3D grid of estimated CO2 saturation
        """
        if not self.velocity_diff_maps and not diff_map_name:
            print("Warning: No velocity difference maps available")
            return None
        
        # Use first available diff map if none specified
        if diff_map_name is None:
            diff_map_name = list(self.velocity_diff_maps.keys())[0]
        
        if diff_map_name not in self.velocity_diff_maps:
            print(f"Error: Velocity difference map '{diff_map_name}' not found")
            return None
        
        # Get velocity difference data
        diff_map = self.velocity_diff_maps[diff_map_name]['data']
        
        # Normalize extreme velocity differences if needed
        diff_map = self.normalize_velocity_differences(diff_map)
        
        # Default parameters for conversion
        if params is None:
            params = {
                'min_vel_diff': -200,  # Minimum velocity difference (m/s)
                'max_vel_diff': 0,     # Maximum velocity difference (m/s)
                'min_saturation': 0,   # Minimum saturation (fraction)
                'max_saturation': 1,   # Maximum saturation (fraction)
                'threshold': -20       # Threshold for considering CO2 presence (m/s)
            }
        
        # Create saturation grid with same dimensions as binary grid
        if self.binary_grid is None:
            print("Warning: Binary grid not created. Call create_binary_plume_grid() first.")
            return None
        
        nx, ny, nz = self.binary_grid.shape
        saturation_grid = np.zeros((nx, ny, nz))
        
        # Convert velocity difference to saturation
        print(f"Converting velocity differences to CO2 saturation using {method} method")
        
        if method == 'linear':
            # Linear scaling between velocity difference and saturation
            # More negative velocity difference = higher CO2 saturation
            
            # Create normalized diff map (invert since more negative = more CO2)
            norm_diff = np.zeros_like(diff_map)
            mask = diff_map <= params['threshold']
            
            # Apply only where velocity difference is below threshold
            norm_diff[mask] = np.clip(
                (diff_map[mask] - params['min_vel_diff']) / 
                (params['max_vel_diff'] - params['min_vel_diff']),
                0, 1
            )
            
            # Invert (1 = high saturation, 0 = no saturation)
            norm_diff[mask] = 1 - norm_diff[mask]
            
            # For each layer, map the 2D velocity difference to saturation
            for layer_idx in range(nz):
                # Check if there's plume data for this layer
                layer_name = f"L{layer_idx+1}"
                if layer_name in self.plume_data and np.sum(self.binary_grid[:, :, layer_idx]) > 0:
                    # Map velocity difference to this layer using binary grid as mask
                    layer_mask = self.binary_grid[:, :, layer_idx] > 0
                    
                    # Resize norm_diff to match grid if needed
                    if norm_diff.shape != layer_mask.shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (layer_mask.shape[0] / norm_diff.shape[0], 
                                        layer_mask.shape[1] / norm_diff.shape[1])
                        resized_diff = zoom(norm_diff, zoom_factors)
                    else:
                        resized_diff = norm_diff
                    
                    # Apply saturation only inside plume
                    saturation_grid[:, :, layer_idx][layer_mask] = (
                        params['min_saturation'] + 
                        resized_diff[layer_mask] * (params['max_saturation'] - params['min_saturation'])
                    )
        
        elif method == 'gassmann':
            # Apply Gassmann fluid substitution equation (simplified)
            # This is kept as an option for advanced users
            
            # Parameters for Gassmann equation
            if 'K_mineral' not in params:
                params.update({
                    'K_mineral': 36.6e9,    # Bulk modulus of mineral (Pa)
                    'K_frame': 2.8e9,       # Bulk modulus of frame (Pa)
                    'K_water': 2.25e9,      # Bulk modulus of water (Pa)
                    'K_co2': 0.025e9,       # Bulk modulus of CO2 (Pa)
                    'rho_mineral': 2650,    # Density of mineral (kg/m³)
                    'rho_water': 1030,      # Density of water (kg/m³)
                    'rho_co2': 700,         # Density of CO2 (kg/m³)
                    'porosity': 0.36        # Porosity (fraction)
                })
            
            # Get baseline velocity (assuming post - pre difference)
            pre_map_name = self.velocity_diff_maps[diff_map_name]['pre_map']
            baseline_vel = self.velocity_maps[pre_map_name]['data']
            
            # Create lookup table of velocity differences for saturation values
            saturation_values = np.linspace(0, 1, 101)
            velocity_diffs = np.zeros_like(saturation_values)
            
            # Gassmann calculations (simplified and vectorized)
            sat = saturation_values
            
            # Calculate effective fluid properties
            K_fluid = 1 / ((sat / params['K_co2']) + ((1 - sat) / params['K_water']))
            K_fluid[sat == 1] = params['K_co2']  # Handle 100% CO2 case
            
            # Calculate mixed density
            rho_mixed = (1 - params['porosity']) * params['rho_mineral'] + \
                        params['porosity'] * ((1 - sat) * params['rho_water'] + sat * params['rho_co2'])
            
            # Calculate saturated bulk modulus
            K_sat = params['K_frame'] + (
                (1 - params['K_frame'] / params['K_mineral'])**2 /
                ((params['porosity'] / K_fluid) + 
                    ((1 - params['porosity']) / params['K_mineral']) - 
                    (params['K_frame'] / params['K_mineral']**2))
            )
            
            # Calculate velocity
            vel_sat = np.sqrt(K_sat / rho_mixed)
            
            # Calculate velocity differences
            v_baseline = np.sqrt(params['K_frame'] / ((1 - params['porosity']) * params['rho_mineral'] + 
                                                    params['porosity'] * params['rho_water']))
            
            velocity_diffs = vel_sat - v_baseline
            
            # Create interpolation function
            from scipy.interpolate import interp1d
            vel_to_sat = interp1d(
                velocity_diffs, 
                saturation_values, 
                bounds_error=False, 
                fill_value=(0, 1)
            )
            
            # Process each layer
            for layer_idx in range(nz):
                # Check if there's plume data for this layer
                layer_name = f"L{layer_idx+1}"
                if layer_name in self.plume_data and np.sum(self.binary_grid[:, :, layer_idx]) > 0:
                    # Use binary grid as mask
                    layer_mask = self.binary_grid[:, :, layer_idx] > 0
                    
                    # Resize diff_map to match grid if needed
                    if diff_map.shape != layer_mask.shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (layer_mask.shape[0] / diff_map.shape[0], 
                                        layer_mask.shape[1] / diff_map.shape[1])
                        resized_diff = zoom(diff_map, zoom_factors)
                    else:
                        resized_diff = diff_map
                    
                    # Convert velocity diff to saturation
                    saturations = vel_to_sat(resized_diff)
                    
                    # Apply only inside plume
                    saturation_grid[:, :, layer_idx][layer_mask] = saturations[layer_mask]
        
        elif method == 'custom':
            # Custom method for user-defined conversion
            if 'conversion_func' not in params:
                print("Error: No conversion function provided for custom method")
                return None
            
            conversion_func = params['conversion_func']
            
            # Apply to each layer
            for layer_idx in range(nz):
                # Check if there's plume data for this layer
                layer_name = f"L{layer_idx+1}"
                if layer_name in self.plume_data and np.sum(self.binary_grid[:, :, layer_idx]) > 0:
                    # Map velocity difference to this layer using binary grid as mask
                    layer_mask = self.binary_grid[:, :, layer_idx] > 0
                    
                    # Resize diff_map to match grid if needed
                    if diff_map.shape != layer_mask.shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (layer_mask.shape[0] / diff_map.shape[0], 
                                        layer_mask.shape[1] / diff_map.shape[1])
                        resized_diff = zoom(diff_map, zoom_factors)
                    else:
                        resized_diff = diff_map
                    
                    # Apply custom conversion
                    saturations = conversion_func(resized_diff)
                    
                    # Apply only inside plume
                    saturation_grid[:, :, layer_idx][layer_mask] = saturations[layer_mask]
        
        else:
            print(f"Error: Unknown conversion method '{method}'")
            return None
        
        # Store saturation grid
        self.saturation_grid = saturation_grid
        
        print(f"Saturation grid created with range: {np.min(saturation_grid):.3f} to {np.max(saturation_grid):.3f}")
        
        return saturation_grid
    
    def add_directional_features(self, X_features):
        """
        Add directional features derived from X and Y coordinates
        
        Parameters:
        -----------
        X_features : numpy.ndarray
            Feature matrix with X, Y, Z positions in first three columns
            
        Returns:
        --------
        numpy.ndarray
            Updated feature matrix with directional features added
        """
        # Extract X, Y coordinates (assuming first two columns)
        x_coords = X_features[:, 0]
        y_coords = X_features[:, 1]
        
        # Calculate center of the coordinate system (reference point)
        # Using injection point as reference is often meaningful
        if hasattr(self, 'inj_x') and hasattr(self, 'inj_y'):
            # Use injection point if available
            x_center = self.inj_x / self.binary_grid.shape[0]
            y_center = self.inj_y / self.binary_grid.shape[1]
        else:
            # Use mean coordinates as center otherwise
            x_center = np.mean(x_coords)
            y_center = np.mean(y_coords)
        
        # Calculate vectors from center to each point
        x_vec = x_coords - x_center
        y_vec = y_coords - y_center
        
        # Calculate distance from center (magnitude of vector)
        distance = np.sqrt(x_vec**2 + y_vec**2)
        
        # Calculate angle in radians (within 0 to 2π range)
        angle = np.arctan2(y_vec, x_vec)
        
        # Convert to degrees and ensure 0-360 range
        angle_deg = np.degrees(angle) % 360
        
        # For 0-90 degree features, we can use trigonometric functions
        # These capture the directional component while constraining to your desired range
        sin_angle = np.sin(angle) 
        cos_angle = np.cos(angle)
        
        # Calculate quadrant features (one-hot encoded)
        # Each point will have a 1 in exactly one of these features
        q1 = ((angle_deg >= 0) & (angle_deg < 90)).astype(float)
        q2 = ((angle_deg >= 90) & (angle_deg < 180)).astype(float)
        q3 = ((angle_deg >= 180) & (angle_deg < 270)).astype(float)
        q4 = ((angle_deg >= 270) & (angle_deg < 360)).astype(float)
        
        # Add features to the matrix
        enhanced_features = np.column_stack([
            X_features,  # Original features
            distance,    # Distance from center
            sin_angle,   # Y component
            cos_angle,   # X component
            q1, q2, q3, q4  # Quadrant indicators
        ])
        
        return enhanced_features
    
    def prepare_ml_features(self, include_saturation=False):
        """
        Extract features for machine learning with transformed coordinates 
        using cosines of angles at 10-degree increments
        
        Parameters:
        -----------
        include_saturation : bool
            Whether to include saturation as a feature
            
        Returns:
        --------
        tuple
            X_features, y_targets for ML training
        """
        if self.binary_grid is None:
            print("Error: Binary grid not created. Call create_binary_plume_grid() first.")
            return None, None, None
        
        # Extract grid dimensions
        nx, ny, nz = self.binary_grid.shape
        total_cells = nx * ny * nz
        
        # Create coordinate grids
        x_indices, y_indices, z_indices = np.meshgrid(
            np.arange(nx) / nx,
            np.arange(ny) / ny,
            np.arange(nz) / nz,
            indexing='ij'
        )
        
        # Extract target values (CO2 presence)
        y_targets = self.binary_grid.flatten()
        
        # Create base features
        X_features = np.column_stack([
            x_indices.flatten(),  # X position
            y_indices.flatten(),  # Y position
            z_indices.flatten(),  # Z position
        ])
        
        # Add distance from injection point (assumed to be in the center of layer 0)
        inj_x, inj_y, inj_z = nx // 2, ny // 2, 0
        
        # Correct injection point if well position data available
        if self.well_position_data:
            for well_name, pos_df in self.well_position_data.items():
                if "15/9-A-16" in well_name or "injection" in well_name.lower():
                    # Use the first position as injection point
                    try:
                        # Find coordinate bounds from plume data
                        all_coords = []
                        for layer, segments in self.plume_data.items():
                            for seg_id, coords in segments.items():
                                all_coords.append(coords)
                        
                        if all_coords:
                            all_coords = np.vstack(all_coords)
                            x_min, y_min = np.min(all_coords, axis=0)
                            x_max, y_max = np.max(all_coords, axis=0)
                            
                            # Extract X and Y offsets
                            x_offset = pos_df['X_Offset'].iloc[0] if 'X_Offset' in pos_df.columns else 0
                            y_offset = pos_df['Y_Offset'].iloc[0] if 'Y_Offset' in pos_df.columns else 0
                            
                            # Convert to grid indices
                            inj_x = int((x_offset - x_min) / (x_max - x_min) * (nx - 1))
                            inj_y = int((y_offset - y_min) / (y_max - y_min) * (ny - 1))
                            inj_z = 0  # Assume injection at bottom layer
                    except Exception as e:
                        print(f"Could not extract injection point from well data: {e}")
        
        # Calculate distance to injection point
        distances = np.sqrt(
            (x_indices - inj_x/nx)**2 + 
            (y_indices - inj_y/ny)**2 + 
            (z_indices - inj_z/nz)**2
        )
        
        # Add distance to features
        X_features = np.column_stack([X_features, distances.flatten()])
        
        # Get X and Y coordinates relative to injection point
        dx = x_indices - inj_x/nx
        dy = y_indices - inj_y/ny
        
        # Create angle-based features
        directional_features = []
        directional_names = []
        
        # Apply transformations for each 10-degree increment from 0 to 90 degrees
        for angle_deg in range(0, 100, 10):  # 0, 10, 20, ..., 90 degrees
            if angle_deg == 0:
                # At 0 degrees, just use x coordinate (cos(0) = 1, sin(0) = 0)
                transformed_x = dx
                feature_name = 'X_0deg'
            elif angle_deg == 90:
                # At 90 degrees, just use y coordinate (cos(90) = 0, sin(90) = 1)
                transformed_x = dy
                feature_name = 'X_90deg'
            else:
                # For angles in between, apply the cosine transformation
                angle_rad = np.radians(angle_deg)
                # Rotate coordinates by this angle
                transformed_x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                feature_name = f'X_{angle_deg}deg'
            
            directional_features.append(transformed_x.flatten())
            directional_names.append(feature_name)
        
        # Add directional features
        X_features = np.column_stack([X_features, *directional_features])
        
        # Add layer type (1 for sandstone, 0 for shale based on layer number)
        layer_type = np.ones((nx, ny, nz))
        for z in range(nz):
            if z % 2 == 1:  # Every other layer is shale in simplified model
                layer_type[:, :, z] = 0
        
        # Add layer type to features
        X_features = np.column_stack([X_features, layer_type.flatten()])
        
        # Add simplified porosity and permeability
        porosity = np.ones((nx, ny, nz)) * 0.36  # 36% for Utsira Formation
        permeability = np.ones((nx, ny, nz)) * 2000  # 2000 mD for sandstone
        
        # Mark shale layers with low permeability
        for z in range(nz):
            if z % 2 == 1:  # Mark shale layers
                permeability[:, :, z] = 0.001  # 0.001 mD for shale
        
        # Add to features (log transform permeability for better scaling)
        X_features = np.column_stack([
            X_features, 
            porosity.flatten(),
            np.log10(permeability.flatten() + 1e-6)
        ])
        
        # Add saturation as feature if requested and available
        if include_saturation and self.saturation_grid is not None:
            X_features = np.column_stack([X_features, self.saturation_grid.flatten()])
        
        # Add velocity difference as feature if available
        if self.velocity_diff_maps:
            # Use first available diff map
            diff_map_name = list(self.velocity_diff_maps.keys())[0]
            diff_map = self.velocity_diff_maps[diff_map_name]['data']
            
            # Resize to match grid if needed
            if diff_map.shape != (nx, ny):
                from scipy.ndimage import zoom
                zoom_factors = (nx / diff_map.shape[0], ny / diff_map.shape[1])
                resized_diff = zoom(diff_map, zoom_factors)
            else:
                resized_diff = diff_map
            
            # Replicate for each layer and add to features
            diff_feature = np.zeros((nx, ny, nz))
            for z in range(nz):
                diff_feature[:, :, z] = resized_diff
            
            X_features = np.column_stack([X_features, diff_feature.flatten()])
        
        # Create feature names
        feature_names = ['X_Position', 'Y_Position', 'Z_Position', 'Distance_to_Injection']
        feature_names.extend(directional_names)
        feature_names.extend(['Layer_Type', 'Porosity', 'Log_Permeability'])
        
        if include_saturation and self.saturation_grid is not None:
            feature_names.append('CO2_Saturation')
        
        if self.velocity_diff_maps:
            feature_names.append('Velocity_Difference')
        
        print(f"Feature matrix created: {X_features.shape}")
        print(f"Features: {', '.join(feature_names)}")
        
        return X_features, y_targets, feature_names
    
    # def prepare_ml_features(self, include_saturation=False):
    #     """
    #     Extract features for machine learning including discrete directional features
    #     with 10-degree increments (10°, 20°, 30°, etc.)
        
    #     Parameters:
    #     -----------
    #     include_saturation : bool
    #         Whether to include saturation as a feature
            
    #     Returns:
    #     --------
    #     tuple
    #         X_features, y_targets for ML training
    #     """
    #     if self.binary_grid is None:
    #         print("Error: Binary grid not created. Call create_binary_plume_grid() first.")
    #         return None, None, None
        
    #     # Extract grid dimensions
    #     nx, ny, nz = self.binary_grid.shape
    #     total_cells = nx * ny * nz
        
    #     # Create coordinate grids
    #     x_indices, y_indices, z_indices = np.meshgrid(
    #         np.arange(nx) / nx,
    #         np.arange(ny) / ny,
    #         np.arange(nz) / nz,
    #         indexing='ij'
    #     )
        
    #     # Extract target values (CO2 presence)
    #     y_targets = self.binary_grid.flatten()
        
    #     # Create base features
    #     X_features = np.column_stack([
    #         x_indices.flatten(),  # X position
    #         y_indices.flatten(),  # Y position
    #         z_indices.flatten(),  # Z position
    #     ])
        
    #     # Add distance from injection point (assumed to be in the center of layer 0)
    #     inj_x, inj_y, inj_z = nx // 2, ny // 2, 0
        
    #     # Correct injection point if well position data available
    #     if self.well_position_data:
    #         for well_name, pos_df in self.well_position_data.items():
    #             if "15/9-A-16" in well_name or "injection" in well_name.lower():
    #                 # Use the first position as injection point
    #                 try:
    #                     # Find coordinate bounds from plume data
    #                     all_coords = []
    #                     for layer, segments in self.plume_data.items():
    #                         for seg_id, coords in segments.items():
    #                             all_coords.append(coords)
                        
    #                     if all_coords:
    #                         all_coords = np.vstack(all_coords)
    #                         x_min, y_min = np.min(all_coords, axis=0)
    #                         x_max, y_max = np.max(all_coords, axis=0)
                            
    #                         # Extract X and Y offsets
    #                         x_offset = pos_df['X_Offset'].iloc[0] if 'X_Offset' in pos_df.columns else 0
    #                         y_offset = pos_df['Y_Offset'].iloc[0] if 'Y_Offset' in pos_df.columns else 0
                            
    #                         # Convert to grid indices
    #                         inj_x = int((x_offset - x_min) / (x_max - x_min) * (nx - 1))
    #                         inj_y = int((y_offset - y_min) / (y_max - y_min) * (ny - 1))
    #                         inj_z = 0  # Assume injection at bottom layer
    #                 except Exception as e:
    #                         print(f"Could not extract injection point from well data: {e}")
        
    #     # Calculate distance to injection point
    #     distances = np.sqrt(
    #         (x_indices - inj_x/nx)**2 + 
    #         (y_indices - inj_y/ny)**2 + 
    #         (z_indices - inj_z/nz)**2
    #     )
        
    #     # Add distance to features
    #     X_features = np.column_stack([X_features, distances.flatten()])
        
    #     # Add directional features - add vectors for specific angles (10°, 20°, 30°, etc.)
    #     angle_features = []
    #     angle_names = []
        
    #     # Create unit vectors for each 10-degree increment from 0 to 90 degrees
    #     for angle_deg in range(10, 100, 5):  # 10, 20, ..., 90 degrees
    #         angle_rad = np.radians(angle_deg)
            
    #         # Create unit vector components
    #         x_component = np.cos(angle_rad)
    #         y_component = np.sin(angle_rad)
            
    #         # Calculate dot product between the point's direction and this unit vector
    #         # This gives the projection of the point's direction onto this angle
    #         # Vector from injection point to each point
    #         dx = x_indices - inj_x/nx
    #         dy = y_indices - inj_y/ny
            
    #         # Normalize to unit vectors (to focus on direction, not distance)
    #         # Avoid division by zero for the injection point itself
    #         norm = np.sqrt(dx**2 + dy**2)
    #         mask = norm > 0
            
    #         dx_norm = np.zeros_like(dx)
    #         dy_norm = np.zeros_like(dy)
    #         dx_norm[mask] = dx[mask] / norm[mask]
    #         dy_norm[mask] = dy[mask] / norm[mask]
            
    #         # Calculate dot product with angle unit vector
    #         direction_projection = dx_norm * x_component + dy_norm * y_component
            
    #         # Add to features
    #         angle_features.append(direction_projection.flatten())
    #         angle_names.append(f'Direction_{angle_deg}deg')
        
    #     # Add all directional features
    #     X_features = np.column_stack([X_features, *angle_features])
        
    #     # Add layer type (1 for sandstone, 0 for shale based on layer number)
    #     layer_type = np.ones((nx, ny, nz))
    #     for z in range(nz):
    #         if z % 2 == 1:  # Every other layer is shale in simplified model
    #             layer_type[:, :, z] = 0
        
    #     # Add layer type to features
    #     X_features = np.column_stack([X_features, layer_type.flatten()])
        
    #     # Add simplified porosity and permeability
    #     porosity = np.ones((nx, ny, nz)) * 0.36  # 36% for Utsira Formation
    #     permeability = np.ones((nx, ny, nz)) * 2000  # 2000 mD for sandstone
        
    #     # Mark shale layers with low permeability
    #     for z in range(nz):
    #         if z % 2 == 1:  # Mark shale layers
    #             permeability[:, :, z] = 0.001  # 0.001 mD for shale
        
    #     # Add to features (log transform permeability for better scaling)
    #     X_features = np.column_stack([
    #         X_features, 
    #         porosity.flatten(),
    #         np.log10(permeability.flatten() + 1e-6)
    #     ])
        
    #     # Add saturation as feature if requested and available
    #     if include_saturation and self.saturation_grid is not None:
    #         X_features = np.column_stack([X_features, self.saturation_grid.flatten()])
        
    #     # Add velocity difference as feature if available
    #     if self.velocity_diff_maps:
    #         # Use first available diff map
    #         diff_map_name = list(self.velocity_diff_maps.keys())[0]
    #         diff_map = self.velocity_diff_maps[diff_map_name]['data']
            
    #         # Resize to match grid if needed
    #         if diff_map.shape != (nx, ny):
    #             from scipy.ndimage import zoom
    #             zoom_factors = (nx / diff_map.shape[0], ny / diff_map.shape[1])
    #             resized_diff = zoom(diff_map, zoom_factors)
    #         else:
    #             resized_diff = diff_map
            
    #         # Replicate for each layer and add to features
    #         diff_feature = np.zeros((nx, ny, nz))
    #         for z in range(nz):
    #             diff_feature[:, :, z] = resized_diff
            
    #         X_features = np.column_stack([X_features, diff_feature.flatten()])
        
    #     # Create feature names
    #     feature_names = ['X_Position', 'Y_Position', 'Z_Position', 'Distance_to_Injection']
    #     feature_names.extend(angle_names)
    #     feature_names.extend(['Layer_Type', 'Porosity', 'Log_Permeability'])
        
    #     if include_saturation and self.saturation_grid is not None:
    #         feature_names.append('CO2_Saturation')
        
    #     if self.velocity_diff_maps:
    #         feature_names.append('Velocity_Difference')
        
    #     print(f"Feature matrix created: {X_features.shape}")
    #     print(f"Features: {', '.join(feature_names)}")
        
    #     return X_features, y_targets, feature_names
    
    def train_saturation_model(self, model_type='regression_forest'):
        """
        Train a machine learning model to predict CO2 saturation
        
        Parameters:
        -----------
        model_type : str
            Type of model to train ('regression_forest', 'neural_network', 'svr')
            
        Returns:
        --------
        tuple
            Trained model, performance metrics, and feature importance
        """
        if self.saturation_grid is None:
            print("Error: Saturation grid not created. Call convert_velocity_to_saturation() first.")
            return None, None, None
        
        # Prepare features
        X_features, _, feature_names = self.prepare_ml_features(include_saturation=False)
        
        if X_features is None:
            return None, None, None
        
        # Use saturation as target
        y_targets = self.saturation_grid.flatten()
        
        # Use only cells with CO2 presence
        mask = self.binary_grid.flatten() > 0
        X_train = X_features[mask]
        y_train = y_targets[mask]
        
        print(f"Training saturation model on {X_train.shape[0]} cells with CO2 presence")
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Train model based on type
        if model_type == 'regression_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            print("Training Random Forest regression model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            print("Model evaluation:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Extract feature importance
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            
            print("Feature importance:")
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {importance:.4f}")
        
        elif model_type == 'neural_network':
            from sklearn.neural_network import MLPRegressor
            
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            print("Training Neural Network regression model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            print("Model evaluation:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Neural networks don't have direct feature importance
            feature_importance = None
        
        elif model_type == 'svr':
            from sklearn.svm import SVR
            
            model = SVR(
                kernel='rbf',
                C=100,
                epsilon=0.2
            )
            
            print("Training Support Vector Regression model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            print("Model evaluation:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # SVR doesn't have direct feature importance
            feature_importance = None
        
        else:
            print(f"Error: Unknown model type '{model_type}'")
            return None, None, None
        
        # Store model
        self.ml_models['saturation'] = {
            'model': model,
            'type': model_type,
            'metrics': metrics,
            'feature_importance': feature_importance
        }
        
        return model, metrics, feature_importance
    
    def predict_saturation(self, time_factor=1.0):
        """
        Predict CO2 saturation using the trained model
        
        Parameters:
        -----------
        time_factor : float
            Time factor to simulate future states (1.0 = current time)
            
        Returns:
        --------
        numpy.ndarray
            Predicted saturation grid
        """
        if 'saturation' not in self.ml_models:
            print("Error: Saturation model not trained. Call train_saturation_model() first.")
            return None
        
        # Get model
        model = self.ml_models['saturation']['model']
        
        # Prepare features
        X_features, _, _ = self.prepare_ml_features(include_saturation=False)
        
        if X_features is None:
            return None
        
        print(f"Predicting CO2 saturation with time factor {time_factor}...")
        
        # Apply time factor to features
        if time_factor != 1.0:
            # Modify distance to injection to simulate expansion over time
            distance_idx = 3  # Assuming Distance_to_Injection is 4th feature
            X_features[:, distance_idx] = X_features[:, distance_idx] / time_factor
        
        # Make prediction
        y_pred = model.predict(X_features)
        
        # Reshape to grid
        nx, ny, nz = self.binary_grid.shape
        predicted_saturation = y_pred.reshape((nx, ny, nz))
        
        # Apply physics constraints (no saturation in shale layers)
        for z in range(nz):
            if z % 2 == 1:  # Shale layers
                predicted_saturation[:, :, z] = 0
        
        # Apply threshold to get meaningful saturation range
        predicted_saturation = np.clip(predicted_saturation, 0, 1)
        
        print(f"Prediction range: {np.min(predicted_saturation):.3f} to {np.max(predicted_saturation):.3f}")
        
        return predicted_saturation
    

    
    def visualize_layer(self, layer_idx, data_type='saturation', save_path=None, clip_to_plume=False):
        """
        Visualize a specific layer with option to clip to plume boundaries
        
        Parameters:
        -----------
        layer_idx : int
            Layer index (0-based)
        data_type : str
            Type of data to visualize ('binary', 'saturation', 'prediction')
        save_path : str, optional
            Path to save the figure
        clip_to_plume : bool, optional
            Whether to clip the visualization to plume boundaries (default: False)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Select data to visualize
        if data_type == 'binary' and self.binary_grid is not None:
            data = self.binary_grid[:, :, layer_idx].T
            title = f'CO2 Plume Presence - Layer {layer_idx+1}'
            cmap = 'cividis'
            vmin, vmax = 0, 1
        elif data_type == 'saturation' and self.saturation_grid is not None:
            data = self.saturation_grid[:, :, layer_idx].T
            title = f'CO2 Saturation - Layer {layer_idx+1}'
            cmap = 'viridis'
            vmin, vmax = 0, 1
        elif data_type == 'prediction' and 'predicted_saturation' in self.results:
            data = self.results['predicted_saturation'][:, :, layer_idx].T
            title = f'Predicted CO2 Saturation - Layer {layer_idx+1}'
            cmap = 'viridis'
            vmin, vmax = 0, 1
        else:
            print(f"Error: No {data_type} data available for visualization")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show data
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=data_type.capitalize())
        
        # Add title and labels
        if clip_to_plume:
            title += " (Clipped)"
        ax.set_title(title)
        ax.set_xlabel('X Grid Index')
        ax.set_ylabel('Y Grid Index')
        
        # Add plume outlines if possible
        layer_key = f'L{layer_idx+1}'
        if layer_key in self.plume_data:
            # Find coordinate bounds
            all_coords = []
            for layer, segments in self.plume_data.items():
                for seg_id, coords in segments.items():
                    all_coords.append(coords)
            
            all_coords = np.vstack(all_coords)
            x_min, y_min = np.min(all_coords, axis=0)
            x_max, y_max = np.max(all_coords, axis=0)
            
            # Plot each segment
            for seg_id, coords in self.plume_data[layer_key].items():
                # Convert to grid indices
                nx, ny = data.shape[1], data.shape[0]  # Transposed
                x_indices = np.round((coords[:, 0] - x_min) / (x_max - x_min) * (nx - 1)).astype(int)
                y_indices = np.round((coords[:, 1] - y_min) / (y_max - y_min) * (ny - 1)).astype(int)
                ax.plot(x_indices, y_indices, 'k-', linewidth=1)
            
            # Clip to plume boundary if requested
            if clip_to_plume:
                # Create a mask for the current layer
                mask = self.binary_grid[:, :, layer_idx].T > 0
                
                # Set values outside plume to NaN
                masked_data = data.copy()
                masked_data[~mask] = np.nan
                
                # Update image data
                im.set_data(masked_data)
                
                # Adjust extent to focus on plume area
                if np.any(mask):
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
                        # Add padding
                        padding = 5
                        x_min = max(0, x_min - padding)
                        x_max = min(nx - 1, x_max + padding)
                        y_min = max(0, y_min - padding)
                        y_max = min(ny - 1, y_max + padding)
                        
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_saturation(self, data_type='saturation', save_path=None, clip_to_plume=False):
        """
        Create 3D visualization of CO2 saturation with option to clip to plume boundaries
        
        Parameters:
        -----------
        data_type : str
            Type of data to visualize ('binary', 'saturation', 'prediction')
        save_path : str, optional
            Path to save the figure
        clip_to_plume : bool, optional
            Whether to clip the visualization to plume boundaries (default: False)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Select data to visualize
        if data_type == 'binary' and self.binary_grid is not None:
            data = self.binary_grid
            title = 'CO2 Plume Presence'
            threshold = 0.5
        elif data_type == 'saturation' and self.saturation_grid is not None:
            data = self.saturation_grid
            title = 'CO2 Saturation'
            threshold = 0.1  # Minimum saturation to show
        elif data_type == 'prediction' and 'predicted_saturation' in self.results:
            data = self.results['predicted_saturation']
            title = 'Predicted CO2 Saturation'
            threshold = 0.1
        else:
            print(f"Error: No {data_type} data available for visualization")
            return None
        
        # Apply plume clipping if requested
        mask = None
        if clip_to_plume and self.binary_grid is not None:
            mask = self.binary_grid > 0
            if data_type != 'binary':  # Don't mask binary data
                masked_data = data.copy()
                masked_data[~mask] = 0
                data = masked_data
            title += " (Clipped to Plume)"
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get indices and values above threshold
        x_indices, y_indices, z_indices = np.where(data > threshold)
        values = data[x_indices, y_indices, z_indices]
        
        # Create color map
        if data_type == 'binary':
            colors = np.ones_like(values)
            cmap = 'cividis'
            label = 'Presence'
        else:
            colors = values
            cmap = 'viridis'
            label = 'Saturation'
        
        # Plot data points
        scatter = ax.scatter(x_indices, y_indices, z_indices, 
                            c=colors, cmap=cmap, 
                            alpha=0.7, s=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=label)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('X Grid Index')
        ax.set_ylabel('Y Grid Index')
        ax.set_zlabel('Layer')
    
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_time_evolution(self, time_factors=[1.0, 1.5, 2.0], save_dir=None):
        """
        Visualize CO2 saturation evolution over time
        
        Parameters:
        -----------
        time_factors : list
            List of time factors to simulate
        save_dir : str, optional
            Directory to save figures
            
        Returns:
        --------
        list
            List of figure objects
        """
        if 'saturation' not in self.ml_models:
            print("Error: Saturation model not trained. Call train_saturation_model() first.")
            return None
        
        print(f"Visualizing time evolution with factors: {time_factors}")
        
        # Create figures for each time factor
        figures = []
        
        for time_factor in time_factors:
            # Predict saturation at this time
            predicted_saturation = self.predict_saturation(time_factor)
            
            if predicted_saturation is None:
                continue
            
            # Apply physics constraints
            # constrained_saturation = self.apply_physics_constraints(predicted_saturation)

            constrained_saturation = predicted_saturation.copy()
            
            # Store in results
            self.results[f'predicted_saturation_t{time_factor}'] = constrained_saturation
            
            # Create 3D visualization
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get indices and values above threshold
            threshold = 0.1
            x_indices, y_indices, z_indices = np.where(constrained_saturation > threshold)
            values = constrained_saturation[x_indices, y_indices, z_indices]
            
            # Plot data points
            scatter = ax.scatter(x_indices, y_indices, z_indices, 
                                c=values, cmap='viridis', 
                                alpha=0.7, s=5)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Saturation')
            
            # Add title and labels
            ax.set_title(f'Predicted CO2 Saturation - Time Factor {time_factor}')
            ax.set_xlabel('X Grid Index')
            ax.set_ylabel('Y Grid Index')
            ax.set_zlabel('Layer')
            
            # Save figure if requested
            if save_dir:
                save_path = os.path.join(save_dir, f'saturation_evolution_t{time_factor}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            figures.append(fig)
        
        return figures
    

    def calculate_plume_metrics(self, prediction=None):
        """
        Calculate metrics for CO2 plume with improved error handling
        
        Parameters:
        -----------
        prediction : numpy.ndarray, optional
            Predicted saturation grid. If None, use actual saturation.
            
        Returns:
        --------
        dict
            Dictionary of plume metrics
        """
        # Select data to analyze
        if prediction is not None:
            data = prediction
            data_type = 'prediction'
        elif self.saturation_grid is not None:
            data = self.saturation_grid
            data_type = 'saturation'
        elif self.binary_grid is not None:
            data = self.binary_grid
            data_type = 'binary'
        else:
            print("Error: No data available for metric calculation")
            return None
        
        print(f"Calculating plume metrics for {data_type} data...")
        
        # Initialize metrics with zero values for all expected keys
        metrics = {
            'plume_cells': 0,
            'plume_volume_m3': 0,
            'pore_volume_m3': 0,
            'co2_volume_m3': 0,
            'co2_mass_tonnes': 0,
            'x_extent_cells': 0,
            'y_extent_cells': 0,
            'x_extent_m': 0,
            'y_extent_m': 0,
            'area_m2': 0,
            'layer_distribution': {}
        }
        
        # Total plume volume
        cell_volume = 50 * 50 * 2  # m³ (assuming 50m x 50m x 2m cells)
        
        if data_type == 'binary':
            # Count cells with CO2 presence
            metrics['plume_cells'] = np.sum(data > 0.5)
            metrics['plume_volume_m3'] = metrics['plume_cells'] * cell_volume
            
            # Assuming porosity of 0.36
            metrics['pore_volume_m3'] = metrics['plume_volume_m3'] * 0.36
            
            # Assuming average saturation of 0.5
            metrics['co2_volume_m3'] = metrics['pore_volume_m3'] * 0.5
            
            # Assuming CO2 density of 700 kg/m³
            metrics['co2_mass_tonnes'] = metrics['co2_volume_m3'] * 700 / 1000
        else:
            # Sum saturations to get total volume
            metrics['plume_cells'] = np.sum(data > 0.1)  # Cells with significant saturation
            metrics['plume_volume_m3'] = metrics['plume_cells'] * cell_volume
            
            # Calculate pore volume
            metrics['pore_volume_m3'] = metrics['plume_volume_m3'] * 0.36
            
            # Calculate CO2 volume using actual saturations
            co2_volume = 0
            for sat_value in np.unique(data):
                if sat_value > 0.1:  # Significant saturation
                    cell_count = np.sum(data == sat_value)
                    co2_volume += cell_count * cell_volume * 0.36 * sat_value
            
            metrics['co2_volume_m3'] = co2_volume
            
            # Calculate CO2 mass
            metrics['co2_mass_tonnes'] = metrics['co2_volume_m3'] * 700 / 1000
        
        # Layer distribution (percentage in each layer)
        for z in range(data.shape[2]):
            if data_type == 'binary':
                layer_count = np.sum(data[:, :, z] > 0.5)
            else:
                layer_count = np.sum(data[:, :, z] > 0.1)
            
            if metrics['plume_cells'] > 0:
                layer_pct = (layer_count / metrics['plume_cells']) * 100
            else:
                layer_pct = 0
            
            metrics['layer_distribution'][f'L{z+1}'] = {
                'cells': int(layer_count),
                'percentage': round(layer_pct, 2)
            }
        
        # Plume extent
        if metrics['plume_cells'] > 0:
            try:
                x_indices, y_indices, _ = np.where(data > 0.1)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    
                    metrics['x_extent_cells'] = x_max - x_min + 1
                    metrics['y_extent_cells'] = y_max - y_min + 1
                    metrics['x_extent_m'] = metrics['x_extent_cells'] * 50
                    metrics['y_extent_m'] = metrics['y_extent_cells'] * 50
                    metrics['area_m2'] = metrics['x_extent_m'] * metrics['y_extent_m']
            except Exception as e:
                print(f"Error calculating plume extent: {e}")
        
        return metrics

    def visualize_metrics(self, metrics=None, save_path=None):
        """
        Visualize plume metrics with improved error handling
        
        Parameters:
        -----------
        metrics : dict, optional
            Metrics dictionary. If None, calculate metrics.
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if metrics is None:
            metrics = self.calculate_plume_metrics()
        
        if metrics is None:
            print("Warning: No metrics available for visualization")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Ensure all required keys exist in metrics dictionary
        required_keys = ['plume_cells', 'plume_volume_m3', 'pore_volume_m3', 
                        'co2_volume_m3', 'co2_mass_tonnes', 
                        'x_extent_m', 'y_extent_m', 'area_m2']
        
        # Check if any required keys are missing
        missing_keys = [key for key in required_keys if key not in metrics]
        if missing_keys:
            print(f"Warning: Missing required keys in metrics: {missing_keys}")
            # Provide default values for missing keys
            for key in missing_keys:
                metrics[key] = 0
        
        # Handle missing layer_distribution
        if 'layer_distribution' not in metrics:
            print("Warning: layer_distribution not found in metrics, creating empty distribution")
            # Create an empty layer distribution
            metrics['layer_distribution'] = {}
            
            # If we have a binary grid, try to create layer distribution
            if self.binary_grid is not None:
                nx, ny, nz = self.binary_grid.shape
                for z in range(nz):
                    layer_count = np.sum(self.binary_grid[:, :, z] > 0.5)
                    if metrics['plume_cells'] > 0:
                        layer_pct = (layer_count / metrics['plume_cells']) * 100
                    else:
                        layer_pct = 0
                    
                    metrics['layer_distribution'][f'L{z+1}'] = {
                        'cells': int(layer_count),
                        'percentage': round(layer_pct, 2)
                    }
        
        # Plot 1: Layer distribution
        ax1 = fig.add_subplot(221)
        layer_names = []
        layer_pcts = []
        
        for layer, data in metrics['layer_distribution'].items():
            if isinstance(data, dict) and 'cells' in data and data['cells'] > 0:
                layer_names.append(layer)
                layer_pcts.append(data['percentage'])
        
        if layer_names:
            ax1.bar(layer_names, layer_pcts)
            ax1.set_title('CO2 Distribution by Layer')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Percentage (%)')
            ax1.grid(True, linestyle='--', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, 'No layer distribution data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Volume and mass
        ax2 = fig.add_subplot(222)
        labels = ['Plume Volume', 'Pore Volume', 'CO2 Volume']
        values = [
            metrics['plume_volume_m3'] / 1e6,  # Convert to million m³
            metrics['pore_volume_m3'] / 1e6,
            metrics['co2_volume_m3'] / 1e6
        ]
        
        ax2.bar(labels, values)
        ax2.set_title('Volume Metrics')
        ax2.set_ylabel('Million m³')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add CO2 mass
        ax2b = ax2.twinx()
        ax2b.bar(['CO2 Mass'], [metrics['co2_mass_tonnes'] / 1e6], color='red', alpha=0.5)
        ax2b.set_ylabel('Million Tonnes', color='red')
        ax2b.tick_params(axis='y', colors='red')
        
        # Plot 3: Spatial extent
        ax3 = fig.add_subplot(223)
        extent_labels = ['X Extent', 'Y Extent']
        extent_values = [metrics['x_extent_m'] / 1000, metrics['y_extent_m'] / 1000]  # Convert to km
        
        ax3.bar(extent_labels, extent_values)
        ax3.set_title('Plume Extent')
        ax3.set_ylabel('Kilometers')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add area
        ax3b = ax3.twinx()
        ax3b.bar(['Area'], [metrics['area_m2'] / 1e6], color='green', alpha=0.5)  # Convert to km²
        ax3b.set_ylabel('Square Kilometers', color='green')
        ax3b.tick_params(axis='y', colors='green')
        
        # Plot 4: Summary text
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        summary_text = (
            f"CO2 Plume Summary\n"
            f"=================\n\n"
            f"Total cells with CO2: {metrics['plume_cells']}\n"
            f"Total plume volume: {metrics['plume_volume_m3'] / 1e6:.2f} million m³\n"
            f"Total CO2 mass: {metrics['co2_mass_tonnes'] / 1e6:.2f} million tonnes\n\n"
            f"Plume extent: {metrics['x_extent_m'] / 1000:.2f} km × {metrics['y_extent_m'] / 1000:.2f} km\n"
            f"Plume area: {metrics['area_m2'] / 1e6:.2f} km²\n"
        )
        
        # Add highest concentration layer if available
        if layer_names:
            highest_layer = max(metrics['layer_distribution'].items(), 
                            key=lambda x: x[1]['percentage'] if isinstance(x[1], dict) else 0)
            if isinstance(highest_layer[1], dict):
                summary_text += f"\nHighest concentration in layer: {highest_layer[0]}"
        
        ax4.text(0.05, 0.95, summary_text, verticalalignment='top', fontfamily='monospace')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    


    def analyze_layer_velocities(self, layer_name, pre_injection_map_name, post_injection_map_name):
        """
        Analyze velocity differences for a specific layer with improved normalization
        
        Parameters:
        -----------
        layer_name : str
            Name of the layer to analyze (e.g., 'L9')
        pre_injection_map_name : str
            Name of the pre-injection velocity map
        post_injection_map_name : str
            Name of the post-injection velocity map for this layer
            
        Returns:
        --------
        tuple
            Layer velocity difference map, estimated CO2 saturation
        """
        print(f"Analyzing {layer_name} velocities using {post_injection_map_name} as reference...")
        
        # Get velocity maps
        if pre_injection_map_name not in self.velocity_maps:
            print(f"Error: Pre-injection map '{pre_injection_map_name}' not found")
            return None, None
        
        if post_injection_map_name not in self.velocity_maps:
            print(f"Error: Post-injection map '{post_injection_map_name}' not found")
            return None, None
        
        pre_map_data = self.velocity_maps[pre_injection_map_name]['data']
        post_map_data = self.velocity_maps[post_injection_map_name]['data']
        
        # Get map dimensions
        pre_shape = pre_map_data.shape
        post_shape = post_map_data.shape
        
        print(f"  Pre-injection map shape: {pre_shape}")
        print(f"  Post-injection map shape: {post_shape}")
        
        # Resize maps to match dimensions if needed
        from scipy.ndimage import zoom
        
        if pre_shape != post_shape:
            print("  Resizing maps to match dimensions...")
            
            # Resize post to match pre (preserves pre-injection resolution)
            zoom_factors = (pre_shape[0] / post_shape[0], pre_shape[1] / post_shape[1])
            post_map_resized = zoom(post_map_data, zoom_factors)
            pre_map_resized = pre_map_data
            print(f"  Resized post-injection map to {post_map_resized.shape}")
        else:
            pre_map_resized = pre_map_data
            post_map_resized = post_map_data
        
        # Create a mask for this layer based on post-injection map
        layer_mask = ~np.isnan(post_map_resized)
        if not np.any(layer_mask):
            # If no explicit NaNs, try to detect valid data range
            valid_range = (post_map_resized > 1000) & (post_map_resized < 5000)
            if np.any(valid_range):
                layer_mask = valid_range
        
        # Apply layer mask to both maps
        pre_layer = np.where(layer_mask, pre_map_resized, np.nan)
        post_layer = np.where(layer_mask, post_map_resized, np.nan)
        
        # Calculate layer velocity difference
        layer_vel_diff = post_layer - pre_layer
        
        # Normalize extreme velocity differences
        layer_vel_diff = self.normalize_velocity_differences(layer_vel_diff)
        
        # Convert velocity difference to CO2 saturation
        # Define parameters for velocity-to-saturation conversion
        params = {
            'min_vel_diff': -200,  # Minimum velocity difference (m/s)
            'max_vel_diff': 0,    # Maximum velocity difference (m/s)
            'min_saturation': 0,  # Minimum saturation (fraction)
            'max_saturation': 1,  # Maximum saturation (fraction)
            'threshold': -20      # Threshold for considering CO2 presence (m/s)
        }
        
        # Initialize saturation map
        layer_saturation = np.zeros_like(layer_vel_diff)
        
        # Apply threshold to identify CO2 presence
        co2_present = layer_vel_diff <= params['threshold']
        
        # Calculate saturation using linear scaling
        # More negative velocity difference = higher saturation
        if np.any(co2_present):
            # Normalize and invert velocity differences
            norm_diff = np.clip(
                (layer_vel_diff[co2_present] - params['min_vel_diff']) / 
                (params['max_vel_diff'] - params['min_vel_diff']),
                0, 1
            )
            
            # Invert (1 = high saturation, 0 = no saturation)
            layer_saturation[co2_present] = 1 - norm_diff
        
        # Print statistics
        valid_diff = layer_vel_diff[~np.isnan(layer_vel_diff)]
        valid_sat = layer_saturation[~np.isnan(layer_saturation)]
        
        if len(valid_diff) > 0:
            print(f"  {layer_name} velocity difference statistics:")
            print(f"    Min: {np.min(valid_diff):.2f} m/s, Max: {np.max(valid_diff):.2f} m/s, Mean: {np.mean(valid_diff):.2f} m/s")
            print(f"  {layer_name} estimated CO2 saturation:")
            print(f"    Min: {np.min(valid_sat):.3f}, Max: {np.max(valid_sat):.3f}, Mean: {np.mean(valid_sat):.3f}")
            print(f"    CO2 presence detected in {np.sum(co2_present)} cells out of {np.sum(layer_mask)} cells in {layer_name}")
        else:
            print(f"  No valid velocity differences found in {layer_name}")
        
        # Store results in the processor
        if layer_name not in self.results:
            self.results[layer_name] = {}
        
        self.results[layer_name]['vel_diff'] = layer_vel_diff
        self.results[layer_name]['saturation'] = layer_saturation
        self.results[layer_name]['statistics'] = {
            'min_diff': np.min(valid_diff) if len(valid_diff) > 0 else None,
            'max_diff': np.max(valid_diff) if len(valid_diff) > 0 else None,
            'mean_diff': np.mean(valid_diff) if len(valid_diff) > 0 else None,
            'min_sat': np.min(valid_sat) if len(valid_sat) > 0 else None,
            'max_sat': np.max(valid_sat) if len(valid_sat) > 0 else None,
            'mean_sat': np.mean(valid_sat) if len(valid_sat) > 0 else None,
            'co2_cells': np.sum(co2_present) if len(valid_diff) > 0 else 0,
            'total_cells': np.sum(layer_mask) if len(valid_diff) > 0 else 0
        }
        
        return layer_vel_diff, layer_saturation
    
    def visualize_layer9_comparison(self, save_path=None):
        """
        Create a special visualization comparing velocity differences and saturation for Layer 9
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        
        if 'L9' not in self.results or 'vel_diff' not in self.results['L9']:
            print("Error: No Layer 9 analysis results available")
            return None
        
        # Get Layer 9 data
        vel_diff = self.results['L9']['vel_diff']
        saturation = self.results['L9'].get('saturation')
        
        # Create a figure with 3 subplots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Velocity difference
        ax1 = plt.subplot(131)
        
        # Get min/max values and create a centered colormap
        vel_min = np.nanmin(vel_diff)
        vel_max = np.nanmax(vel_diff)
        vel_abs_max = max(abs(vel_min), abs(vel_max))
        vel_norm = plt.Normalize(-vel_abs_max, vel_abs_max)
        
        im1 = ax1.imshow(vel_diff, cmap='RdBu_r', norm=vel_norm, origin='lower')
        plt.colorbar(im1, ax=ax1, label='Velocity Difference (m/s)')
        ax1.set_title('L9 Velocity Difference')
        ax1.set_xlabel('X Grid Index')
        ax1.set_ylabel('Y Grid Index')
        
        # Plot 2: Saturation from velocity diff
        ax2 = plt.subplot(132)
        if saturation is not None:
            im2 = ax2.imshow(saturation, cmap='viridis', vmin=0, vmax=1, origin='lower')
            plt.colorbar(im2, ax=ax2, label='CO2 Saturation')
            ax2.set_title('L9 Estimated CO2 Saturation')
            ax2.set_xlabel('X Grid Index')
            ax2.set_ylabel('Y Grid Index')
        else:
            ax2.text(0.5, 0.5, 'No saturation data available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Binary plume presence
        ax3 = plt.subplot(133)
        if self.binary_grid is not None:
            layer_idx = 8  # L9 is index 8 (0-based)
            binary_data = self.binary_grid[:, :, layer_idx].T
            im3 = ax3.imshow(binary_data, cmap='Reds', vmin=0, vmax=1, origin='lower')
            plt.colorbar(im3, ax=ax3, label='CO2 Presence')
            ax3.set_title('L9 CO2 Plume Presence')
            ax3.set_xlabel('X Grid Index')
            ax3.set_ylabel('Y Grid Index')
            
            # Add plume outlines if available
            if 'L9' in self.plume_data:
                # Find coordinate bounds
                all_coords = []
                for layer, segments in self.plume_data.items():
                    for seg_id, coords in segments.items():
                        all_coords.append(coords)
                
                if all_coords:
                    all_coords = np.vstack(all_coords)
                    x_min, y_min = np.min(all_coords, axis=0)
                    x_max, y_max = np.max(all_coords, axis=0)
                    
                    # Plot each segment
                    for seg_id, coords in self.plume_data['L9'].items():
                        # Convert to grid indices
                        nx, ny = binary_data.shape[1], binary_data.shape[0]  # Transposed
                        x_indices = np.round((coords[:, 0] - x_min) / (x_max - x_min) * (nx - 1)).astype(int)
                        y_indices = np.round((coords[:, 1] - y_min) / (y_max - y_min) * (ny - 1)).astype(int)
                        ax3.plot(x_indices, y_indices, 'k-', linewidth=1)
        else:
            ax3.text(0.5, 0.5, 'No binary plume data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Add statistics as text if available
        if 'L9' in self.results and 'statistics' in self.results['L9']:
            stats = self.results['L9']['statistics']
            
            # Format statistics text
            stats_text = "L9 Statistics:\n"
            if stats.get('mean_diff') is not None:
                stats_text += f"Mean vel diff: {stats['mean_diff']:.2f} m/s\n"
            if stats.get('mean_sat') is not None:
                stats_text += f"Mean saturation: {stats['mean_sat']:.3f}\n"
            if stats.get('co2_cells') is not None:
                stats_text += f"CO2 cells: {stats['co2_cells']} / {stats['total_cells']}\n"
            
            # Add text box to figure
            plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer 9 comparison saved to {save_path}")
        
        return fig
    
    def generate_summary_report(self, output_dir, results, include_ml=True):
        """
        Generate a summary report of the simulation
        
        Parameters:
        -----------
        output_dir : str
            Directory to save report
        results : dict
            Results dictionary
        include_ml : bool
            Whether ML components were included
        """
        import os
        
        report_path = os.path.join(output_dir, 'simulation_summary.txt')
        with open(report_path, 'w') as f:
            f.write("Sleipner CO2 Storage Simulation Summary\n")
            f.write("=======================================\n\n")
            
            # General information
            f.write("Input Data:\n")
            f.write(f"  Plume data: {len(self.plume_data)} layers\n")
            f.write(f"  Grid dimensions: {self.binary_grid.shape if self.binary_grid is not None else 'N/A'}\n")
            f.write(f"  Velocity maps: {len(self.velocity_maps)}\n")
            f.write(f"  Velocity difference maps: {len(self.velocity_diff_maps)}\n\n")
            
            # Model information if ML included
            if include_ml and 'saturation' in self.ml_models:
                model_info = self.ml_models['saturation']
                f.write("Saturation Model:\n")
                f.write(f"  Type: {model_info['type']}\n")
                f.write("  Metrics:\n")
                
                # Check if metrics exist
                if 'metrics' in model_info and model_info['metrics'] is not None:
                    for metric, value in model_info['metrics'].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                else:
                    f.write("    No metrics available\n")
                
                # Write feature importance if available
                feature_importance = model_info.get('feature_importance')
                if feature_importance is not None:
                    f.write("  Feature Importance:\n")
                    for feature, importance in sorted(
                        feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    ):
                        f.write(f"    {feature}: {importance:.4f}\n")
                f.write("\n")
            
            # Current state metrics
            if 'plume_metrics' in results:
                metrics = results['plume_metrics']
                f.write("Current State Metrics:\n")
                
                # Handle potentially missing values
                plume_cells = metrics.get('plume_cells', 0)
                plume_volume = metrics.get('plume_volume_m3', 0) / 1e6
                co2_mass = metrics.get('co2_mass_tonnes', 0) / 1e6
                x_extent = metrics.get('x_extent_m', 0) / 1000
                y_extent = metrics.get('y_extent_m', 0) / 1000
                
                f.write(f"  Plume cells: {plume_cells}\n")
                f.write(f"  Plume volume: {plume_volume:.2f} million m³\n")
                f.write(f"  CO2 mass: {co2_mass:.2f} million tonnes\n")
                f.write(f"  Plume extent: {x_extent:.2f} km × {y_extent:.2f} km\n")
                
                # Layer distribution
                layer_distribution = metrics.get('layer_distribution', {})
                if layer_distribution:
                    f.write("  Layer distribution:\n")
                    for layer, data in layer_distribution.items():
                        if isinstance(data, dict) and data.get('cells', 0) > 0:
                            f.write(f"    {layer}: {data.get('percentage', 0):.2f}% ({data.get('cells', 0)} cells)\n")
                f.write("\n")
            
            # Layer-specific analysis if available
            layer_keys = [k for k in self.results.keys() if k.startswith('L')]
            if layer_keys:
                f.write("Layer-Specific Analysis:\n")
                for layer in sorted(layer_keys):
                    layer_data = self.results[layer]
                    stats = layer_data.get('statistics', {})
                    f.write(f"\n  {layer}:\n")
                    if stats.get('mean_diff') is not None:
                        f.write(f"    Mean velocity difference: {stats.get('mean_diff', 0):.2f} m/s\n")
                    if stats.get('mean_sat') is not None:
                        f.write(f"    Mean CO2 saturation: {stats.get('mean_sat', 0):.3f}\n")
                    if stats.get('co2_cells') is not None:
                        f.write(f"    CO2 presence: {stats.get('co2_cells', 0)} out of {stats.get('total_cells', 0)} cells\n")
                f.write("\n")
            
            # Future predictions if ML included
            future_metrics = [k for k in results.keys() if k.startswith('metrics_t')]
            if include_ml and future_metrics:
                f.write("Future Predictions:\n")
                
                for metrics_key in sorted(future_metrics):
                    # Extract time factor from key
                    time_factor = metrics_key.replace('metrics_t', '')
                    f.write(f"  Time Factor {time_factor}:\n")
                    
                    metrics = results[metrics_key]
                    
                    # Handle potentially missing values
                    plume_cells = metrics.get('plume_cells', 0)
                    plume_volume = metrics.get('plume_volume_m3', 0) / 1e6
                    co2_mass = metrics.get('co2_mass_tonnes', 0) / 1e6
                    x_extent = metrics.get('x_extent_m', 0) / 1000
                    y_extent = metrics.get('y_extent_m', 0) / 1000
                    
                    f.write(f"    Plume cells: {plume_cells}\n")
                    f.write(f"    Plume volume: {plume_volume:.2f} million m³\n")
                    f.write(f"    CO2 mass: {co2_mass:.2f} million tonnes\n")
                    f.write(f"    Plume extent: {x_extent:.2f} km × {y_extent:.2f} km\n")
                    
                    # Calculate change from current state if available
                    if 'plume_metrics' in results:
                        current = results['plume_metrics']
                        current_volume = current.get('plume_volume_m3', 0)
                        current_mass = current.get('co2_mass_tonnes', 0)
                        
                        if current_volume > 0:
                            volume_change = (metrics.get('plume_volume_m3', 0) - current_volume) / current_volume * 100
                            f.write(f"    Volume change: {volume_change:.1f}%\n")
                        
                        if current_mass > 0:
                            mass_change = (metrics.get('co2_mass_tonnes', 0) - current_mass) / current_mass * 100
                            f.write(f"    Mass change: {mass_change:.1f}%\n")
                
                f.write("\n")
            
            f.write("Visualization files have been saved to the output directory.\n")
            print(f"Summary report generated at {report_path}")
    

    def run_simulation(self, output_dir=None, include_ml=True, time_factors=None, clip_to_plume=True):
        """
        Run the complete simulation pipeline with improved visualization options
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save outputs
        include_ml : bool
            Whether to include machine learning and prediction (default: True)
        time_factors : list, optional
            List of time factors for prediction (default: [1.0, 1.5, 2.0])
        clip_to_plume : bool
            Whether to clip visualizations to plume boundaries (default: True)
            
        Returns:
        --------
        dict
            Results dictionary
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Default time factors if not specified
        if time_factors is None:
            time_factors = [1.0, 1.5, 2.0]
        
        results = {}
        
        # 1. Create binary plume grid if not already done
        if self.binary_grid is None and self.plume_data:
            print("Creating binary plume grid...")
            self.create_binary_plume_grid()
        
        if self.binary_grid is None:
            print("Error: No plume data or binary grid available.")
            return results
        
        # 2. Basic metrics and visualization
        print("Calculating plume metrics...")
        try:
            metrics = self.calculate_plume_metrics()
            if metrics is not None:
                results['plume_metrics'] = metrics
        except Exception as e:
            print(f"Error calculating plume metrics: {e}")
        
        # ML and prediction components
        if include_ml:
            # 3. Calculate velocity differences if velocity maps are available
            if self.velocity_maps and not self.velocity_diff_maps and len(self.velocity_maps) >= 2:
                print("Calculating velocity differences...")
                try:
                    # Find pre and post injection maps
                    pre_maps = [name for name, data in self.velocity_maps.items() 
                              if data.get('type') == 'pre-injection']
                    post_maps = [name for name, data in self.velocity_maps.items() 
                               if data.get('type') == 'post-injection']
                    
                    if pre_maps and post_maps:
                        self.calculate_velocity_difference(pre_maps[0], post_maps[0])
                        print(f"Created difference map between {pre_maps[0]} and {post_maps[0]}")
                    else:
                        print("Warning: Could not identify pre- and post-injection maps.")
                except Exception as e:
                    print(f"Error calculating velocity differences: {e}")
            
            # 4. Convert velocity differences to saturation if available
            if self.velocity_diff_maps and self.saturation_grid is None:
                print("Converting velocity differences to saturation...")
                try:
                    self.convert_velocity_to_saturation()
                except Exception as e:
                    print(f"Error converting velocity to saturation: {e}")
            
            # 5. Train saturation model if saturation data available
            if self.saturation_grid is not None and 'saturation' not in self.ml_models:
                print("Training saturation model...")
                try:
                    model, metrics, importance = self.train_saturation_model()
                    if metrics is not None:
                        results['ml_model_metrics'] = metrics
                    if importance is not None:
                        results['feature_importance'] = importance
                except Exception as e:
                    print(f"Error training saturation model: {e}")
            
            # 6. Predict future saturation states
            future_predictions = {}
            
            if 'saturation' in self.ml_models:
                print(f"Predicting saturation for time factors: {time_factors}")
                for factor in time_factors:
                    try:
                        print(f"  Processing time factor: {factor}")
                        predicted_saturation = self.predict_saturation(factor)
                        if predicted_saturation is not None:
                            future_predictions[factor] = predicted_saturation
                            
                            # Store in results
                            results[f'saturation_t{factor}'] = predicted_saturation
                            
                            # Calculate metrics
                            try:
                                pred_metrics = self.calculate_plume_metrics(predicted_saturation)
                                if pred_metrics is not None:
                                    results[f'metrics_t{factor}'] = pred_metrics
                            except Exception as e:
                                print(f"Error calculating metrics for time factor {factor}: {e}")
                    except Exception as e:
                        print(f"Error predicting saturation for time factor {factor}: {e}")
            else:
                print("Warning: No saturation model available for prediction.")
        
        # 7. Layer-specific analysis if velocity maps available
        if self.velocity_maps and len(self.velocity_maps) >= 2:
            print("Performing layer-specific analysis...")
            for layer_idx in range(self.binary_grid.shape[2]):
                layer_name = f"L{layer_idx+1}"
                
                # Check if we have plume data for this layer
                if layer_name in self.plume_data:
                    try:
                        # Find velocity maps for this layer
                        pre_maps = [name for name, data in self.velocity_maps.items() 
                                  if data.get('type') == 'pre-injection']
                        post_maps = [name for name, data in self.velocity_maps.items() 
                                   if data.get('type') == 'post-injection']
                        
                        if pre_maps and post_maps:
                            pre_map = pre_maps[0]
                            post_map = post_maps[0]
                            
                            print(f"  Analyzing {layer_name} with {pre_map} and {post_map}")
                            self.analyze_layer_velocities(layer_name, pre_map, post_map)
                    except Exception as e:
                        print(f"Error analyzing layer {layer_name}: {e}")
        
        # 8. Visualize results
        if output_dir:
            print(f"Saving visualizations to {output_dir}...")
            
            # Visualize each layer of binary plume
            for layer_idx in range(self.binary_grid.shape[2]):
                try:
                    # Binary plume - both clipped and full
                    fig = self.visualize_layer(
                        layer_idx, 
                        data_type='binary', 
                        clip_to_plume=False,
                        save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_binary_full.png')
                    )
                    if fig:
                        plt.close(fig)
                    
                    if clip_to_plume:
                        fig = self.visualize_layer(
                            layer_idx, 
                            data_type='binary', 
                            clip_to_plume=True,
                            save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_binary_clipped.png')
                        )
                        if fig:
                            plt.close(fig)
                    
                    # Saturation visualization if available
                    if self.saturation_grid is not None:
                        fig = self.visualize_layer(
                            layer_idx, 
                            data_type='saturation', 
                            clip_to_plume=False,
                            save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_saturation_full.png')
                        )
                        if fig:
                            plt.close(fig)
                        
                        if clip_to_plume:
                            fig = self.visualize_layer(
                                layer_idx, 
                                data_type='saturation', 
                                clip_to_plume=True,
                                save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_saturation_clipped.png')
                            )
                            if fig:
                                plt.close(fig)
                except Exception as e:
                    print(f"Error visualizing layer {layer_idx+1}: {e}")
            
            # 3D visualization of binary plume
            try:
                fig = self.visualize_3d_saturation(
                    data_type='binary',
                    clip_to_plume=False,
                    save_path=os.path.join(output_dir, '3d_binary_full.png')
                )
                if fig:
                    plt.close(fig)
                
                if clip_to_plume:
                    fig = self.visualize_3d_saturation(
                        data_type='binary',
                        clip_to_plume=True,
                        save_path=os.path.join(output_dir, '3d_binary_clipped.png')
                    )
                    if fig:
                        plt.close(fig)
            except Exception as e:
                print(f"Error creating 3D binary visualization: {e}")
            
            # 3D visualization of saturation if available
            if self.saturation_grid is not None:
                try:
                    fig = self.visualize_3d_saturation(
                        data_type='saturation',
                        clip_to_plume=False,
                        save_path=os.path.join(output_dir, '3d_saturation_full.png')
                    )
                    if fig:
                        plt.close(fig)
                    
                    if clip_to_plume:
                        fig = self.visualize_3d_saturation(
                            data_type='saturation',
                            clip_to_plume=True,
                            save_path=os.path.join(output_dir, '3d_saturation_clipped.png')
                        )
                        if fig:
                            plt.close(fig)
                except Exception as e:
                    print(f"Error creating 3D saturation visualization: {e}")
            
            # Layer-specific comparisons
            for layer_name, layer_data in self.results.items():
                if layer_name.startswith('L') and 'saturation' in layer_data:
                    try:
                        # Create layer-specific comparison visualization
                        if layer_name == 'L9':  # Special visualization for Layer 9
                            fig = self.visualize_layer9_comparison(
                                save_path=os.path.join(output_dir, f'{layer_name}_comparison.png')
                            )
                            if fig:
                                plt.close(fig)
                    except Exception as e:
                        print(f"Error creating comparison for {layer_name}: {e}")
            
            # Future prediction visualizations
            if include_ml and future_predictions:
                for factor, prediction in future_predictions.items():
                    try:
                        # Store result temporarily for visualization
                        self.results['predicted_saturation'] = prediction
                        
                        # Visualize each layer
                        for layer_idx in range(prediction.shape[2]):
                            fig = self.visualize_layer(
                                layer_idx, 
                                data_type='prediction', 
                                clip_to_plume=False,
                                save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_prediction_t{factor}_full.png')
                            )
                            if fig:
                                plt.close(fig)
                            
                            if clip_to_plume:
                                fig = self.visualize_layer(
                                    layer_idx, 
                                    data_type='prediction', 
                                    clip_to_plume=True,
                                    save_path=os.path.join(output_dir, f'layer_{layer_idx+1}_prediction_t{factor}_clipped.png')
                                )
                                if fig:
                                    plt.close(fig)
                        
                        # 3D visualization
                        fig = self.visualize_3d_saturation(
                            data_type='prediction',
                            clip_to_plume=False,
                            save_path=os.path.join(output_dir, f'3d_prediction_t{factor}_full.png')
                        )
                        if fig:
                            plt.close(fig)
                        
                        if clip_to_plume:
                            fig = self.visualize_3d_saturation(
                                data_type='prediction',
                                clip_to_plume=True,
                                save_path=os.path.join(output_dir, f'3d_prediction_t{factor}_clipped.png')
                            )
                            if fig:
                                plt.close(fig)
                    except Exception as e:
                        print(f"Error visualizing predictions for time factor {factor}: {e}")
            
            # Generate summary report
            try:
                self.generate_summary_report(output_dir, results, include_ml)
            except Exception as e:
                print(f"Error generating summary report: {e}")
        
        # Store all results
        self.results.update(results)
        
        return results
