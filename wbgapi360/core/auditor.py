import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataAuditor:
    """
    The 'Eyes' of the Senior Analyst.
    Checks data before it reaches the user.
    """
    
    @staticmethod
    def audit(df: pd.DataFrame) -> dict:
        """
        Performs a full audit on the dataframe.
        Returns a report dictionary.
        """
        if df.empty:
            return {"status": "CRITICAL", "message": "No data returned."}
            
        report = {"status": "OK", "warnings": []}
        
        # 1. Gaps Audit (The "Venezuela Check")
        # Ensure we have column 'OBS_VALUE'
        val_col = 'OBS_VALUE'
        if val_col not in df.columns:
             # Try to find numeric column
             nums = df.select_dtypes(include=['number'])
             if not nums.empty:
                 val_col = nums.columns[0]
        
        if val_col in df.columns:
            missing = df[val_col].isna().sum()
            total = len(df)
            completeness = 1.0 - (missing / total) if total > 0 else 0
            
            if completeness < 0.8:
                report['warnings'].append(f"High data gap detected. Completeness: {completeness:.1%}. Check for missing countries/years.")
                report['status'] = "WARNING"
                
            if completeness == 0:
                 report['status'] = "CRITICAL"
                 report['message'] = "Data is completely empty (Black Hole)."

        # 2. Scale Audit (The "Japan vs Zimbabwe Check")
        # Check variance / magnitude
        if val_col in df.columns and pd.api.types.is_numeric_dtype(df[val_col]):
            vmin = df[val_col].min()
            vmax = df[val_col].max()
            
            if vmax > 0 and vmin > 0:
                ratio = vmax / vmin
                if ratio > 10_000: # 4 orders of magnitude
                    report['warnings'].append(f"Extreme Scale Differences detected (Max/Min ratio: {ratio:,.0f}). Consider Log Scale.")
                    
        return report

    @staticmethod
    def detect_scale_conflict(df: pd.DataFrame) -> dict:
        """
        Analyzes if the dataframe contains mixed series with incompatible scales
        (e.g., Billions vs Percentages).
        Returns a dict with 'has_conflict', 'micro_cols', 'macro_cols'.
        """
        conflict_info = {'has_conflict': False, 'micro_cols': [], 'macro_cols': []}
        
        # Identify numeric columns (excluding Year if it's a column)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c.lower() not in ['year', 'time_period']]
        
        if len(numeric_cols) < 2:
            return conflict_info
            
        micro = []
        macro = []
        
        for col in numeric_cols:
            # Check magnitude (using median to be robust against outliers)
            median_val = df[col].abs().median()
            
            # Logic: Micro < 500 (covers %, indices 0-100, etc)
            #        Macro > 1,000,000 (Millions, Billions)
            if median_val < 500:
                micro.append(col)
            elif median_val > 1_000_000:
                macro.append(col)
                
        # Conflict exists if we have BOTH types
        if micro and macro:
            conflict_info['has_conflict'] = True
            conflict_info['micro_cols'] = micro
            conflict_info['macro_cols'] = macro
            
        return conflict_info

    @staticmethod
    def print_report(report):
        """Prints a user-friendly audit report to console"""
        if report['status'] == 'OK' and not report['warnings']:
            # Silent if perfect? Or subtle nod?
            return 

        print(f"\n[SENIOR ANALYST AUDIT] Status: {report['status']}")
        for w in report.get('warnings', []):
            print(f"  ⚠ {w}")
        if 'message' in report:
            print(f"  ! {report['message']}")
