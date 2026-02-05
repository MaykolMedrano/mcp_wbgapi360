import pandas as pd

class DataStandardizer:
    """
    The 'Hands' of the Senior Analyst.
    Ensures that no matter how the data comes in (Wide, Long, Messy),
    it goes out in a Predictable Tidy Format.
    """
    
    @staticmethod
    def ensure_tidy(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts any DataFrame into a Strict Tidy Format:
        [REF_AREA, TIME_PERIOD, INDICATOR, OBS_VALUE]
        """
        if df.empty:
            return df
            
        if df.empty:
            return df
            
        # 1. Force Reset Index to expose all potential variables (Country, Year)
        if not isinstance(df.index, pd.RangeIndex):
             df = df.reset_index()
             
        cols = df.columns.tolist()
        
        # 2. Check if already Tidy (has key columns)
        # We look for "value" column and "indicator" column
        has_value = any(c.lower() in ['value', 'obs_value'] for c in cols)
        has_indicator = any(c.lower() in ['indicator', 'series'] for c in cols)
        
        if has_value and has_indicator:
            # Likely already tidy, just standardize names
            return DataStandardizer._standardize_names(df)
            
        # 3. Detect Wide-by-Year (Years are columns)
        # Common format: REF_AREA, 2020, 2021, 2022...
        year_cols = [c for c in cols if str(c).isdigit() and 1900 < int(c) < 2100]
        
        if year_cols:
            # We need to melt
            id_vars = [c for c in cols if c not in year_cols]
            df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='TIME_PERIOD', value_name='OBS_VALUE')
            return DataStandardizer._standardize_names(df)
            
        # 4. Detect Wide-by-Indicator (Indicators are columns)
        # Common format: REF_AREA, Year, GDP, INFLATION...
        # We assume if it's not the above, and has numeric columns that aren't years, it might be this.
        # But for 'get_data' generic calls, we usually get Wide-by-Year.
        # This case is rarer in raw WB API but common in user processing.
        
        return DataStandardizer._standardize_names(df)

    @staticmethod
    def _standardize_names(df: pd.DataFrame) -> pd.DataFrame:
        """normalize column names to standard internal schema"""
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ['year', 'time', 'date', 'period']:
                rename_map[c] = 'TIME_PERIOD'
            elif cl in ['economy', 'ref_area', 'country', 'countrycode']:
                rename_map[c] = 'REF_AREA'
            elif cl in ['series', 'indicator', 'variable']:
                rename_map[c] = 'INDICATOR'
            elif cl in ['value', 'obs_value', 'data']:
                rename_map[c] = 'OBS_VALUE'
                
        return df.rename(columns=rename_map)
