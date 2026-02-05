import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import warnings
import numpy as np
from typing import Optional, Tuple

# Default figure sizes (optimized for Colab/Notebooks)
DEFAULT_FIGSIZE = (8, 5)
DEFAULT_FIGSIZE_MAP = (12, 7)
DEFAULT_FIGSIZE_SMALL = (6, 4)

# FT Color Variables
FT_BACKGROUND = "#FFFFFF"
FT_TITLE_COLOR = "#1A1A1A"
FT_SUBTITLE_COLOR = "#6E6E6E"
FT_AXIS_COLOR = "#4D4D4D"
FT_GRID_COLOR = "#E6E6E6"
FT_TICK_COLOR = "#B0B0B0"
FT_SOURCE_COLOR = "#6E6E6E"
FT_BLUE = "#3A5F7D"
FT_GREEN = "#2F7F6F"
FT_RED = "#C44E3B"
FT_MUSTARD = "#C9A227"
FT_PURPLE = "#6B5B95"
FT_NEUTRAL_DARK = "#75787B"
FT_NEUTRAL_LIGHT = "#A6A9AA"
FT_NEUTRAL_LINE = "#8C8C8C"
FT_PALETTE = [FT_BLUE, FT_GREEN, FT_RED, FT_MUSTARD, FT_PURPLE, FT_NEUTRAL_DARK, FT_NEUTRAL_LIGHT]

# Regional Bounding Boxes [lon_min, lat_min, lon_max, lat_max]
REGION_BBOX = {
    'latam': [-120, -60, -30, 35],
    'africa': [-20, -40, 55, 40],
    'europe': [-15, 35, 45, 72],
    'asia': [60, -15, 150, 55],
    'mena': [-20, 10, 65, 45],
    'southasia': [60, 0, 100, 40],
    'eastasia': [95, -10, 150, 50],
    'northamerica': [-170, 15, -50, 75],
    'oceania': [110, -50, 180, 0],
    'caribbean': [-90, 10, -55, 25],
    'centralamerica': [-95, 5, -75, 22],
    'southamerica': [-85, -60, -30, 15],
}

class Visualizer:
    # Reference figsize for default font sizes (FT standard)
    REF_FIGSIZE = (10, 6)
    REF_FONTS = {
        'title': 18,
        'subtitle': 12,
        'axis_label': 10,
        'tick': 9,
        'source': 8,
        'legend': 9,
        'annotation': 9
    }
    
    def __init__(self):
        self._current_figsize = self.REF_FIGSIZE
    
    def _get_scaled_fonts(self, figsize=None):
        """Calculate font sizes scaled proportionally to figsize."""
        fs = figsize or self._current_figsize
        # Scale factor based on width (primary driver of text readability)
        scale = fs[0] / self.REF_FIGSIZE[0]
        # Clamp scale between 0.6 and 1.2 to avoid extreme sizes
        scale = max(0.6, min(1.2, scale))
        
        return {
            'title': int(self.REF_FONTS['title'] * scale),
            'subtitle': int(self.REF_FONTS['subtitle'] * scale),
            'axis_label': int(self.REF_FONTS['axis_label'] * scale),
            'tick': int(self.REF_FONTS['tick'] * scale),
            'source': int(self.REF_FONTS['source'] * scale),
            'legend': int(self.REF_FONTS['legend'] * scale),
            'annotation': int(self.REF_FONTS['annotation'] * scale)
        }

    def _apply_theme_context(self):
        """Configura el estilo FT explícitamente."""
        ft_rc = {
            'figure.facecolor': FT_BACKGROUND,
            'axes.facecolor': FT_BACKGROUND,
            'savefig.facecolor': FT_BACKGROUND,
            
            # Fonts
            'font.family': 'sans-serif',
            'font.sans-serif': ['Metric', 'Arial', 'Helvetica', 'DejaVu Sans'],
            'font.serif': ['Financier Display', 'Georgia', 'Times New Roman', 'DejaVu Serif'],
            'font.size': 12,
            
            # Text Colors
            'text.color': FT_AXIS_COLOR,
            'axes.labelcolor': FT_AXIS_COLOR,
            'xtick.color': FT_TICK_COLOR,
            'ytick.color': FT_TICK_COLOR,
            
            # Spines (Bordes)
            'axes.edgecolor': FT_GRID_COLOR,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': False,
            
            # Grids
            'axes.grid': False,
            'grid.color': FT_GRID_COLOR,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
        }
        
        plt.rcParams.update(ft_rc)
        sns.set_palette(FT_PALETTE)
        
        plt.rcParams['axes.spines.bottom'] = True
        plt.rcParams['axes.edgecolor'] = FT_GRID_COLOR # Subtle

    def _format_axis(self, ax, axis='y'):
        """Applies FT number formatting."""
        def ft_formatter(x, pos):
            if x >= 1e9: return f'{x*1e-9:,.1f}bn'.replace('.0bn', 'bn')
            if x >= 1e6: return f'{x*1e-6:,.1f}m'.replace('.0m', 'm')
            if x >= 1e3: return f'{x*1e-3:,.0f}k'
            if x == 0: return '0'
            return f'{x:,.0f}' if x.is_integer() else f'{x:,.2f}'

        if axis == 'y' or axis == 'both':
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(ft_formatter))
        if axis == 'x' or axis == 'both':
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(ft_formatter))

    def _finalize_chart(self, fig, ax, title, subtitle, save_path, source="World Bank Data360", is_regional_map=False):
        # Get scaled font sizes based on current figsize
        fonts = self._get_scaled_fonts()
        
        # Main title (Serif, Dark, scaled)
        ax.text(x=0, y=1.15, s=title, transform=ax.transAxes, 
                fontsize=fonts['title'], weight='semibold', color=FT_TITLE_COLOR, ha='left',
                fontfamily='serif')
        
        # Subtitle (Sans, Medium grey, scaled)
        if subtitle:
            ax.text(x=0, y=1.06, s=subtitle, transform=ax.transAxes, 
                    fontsize=fonts['subtitle'], color=FT_SUBTITLE_COLOR, ha='left')

        # Source note
        plt.figtext(0.05, 0.01, f"Source: {source}", 
                    fontsize=fonts['source'], color=FT_SOURCE_COLOR, ha='left', va='bottom')
        
        # Scale tick labels
        ax.tick_params(axis='both', labelsize=fonts['tick'])

        
        # Adjust layout based on map type
        if is_regional_map:
            # Regional maps: use subplots_adjust for better centering
            plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.12)
        else:
            # Standard layout
            plt.tight_layout(rect=[0.02, 0.05, 0.95, 0.85]) 
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=FT_BACKGROUND)
            plt.close(fig)
            print(f"[Visual] Chart saved to {save_path}")
        else:
            plt.show()

    def _prepare_data(self, df, auto_rename=False):
        df = df.copy()
        val_label = "Value"
        
        if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
             df = df.reset_index()
        
        for col in ['Indicator Name', 'Series Name', 'indicator_name']:
            if col in df.columns:
                 first_val = df[col].dropna().unique()
                 if len(first_val) > 0:
                     val_label = str(first_val[0])
                     break
        
        if auto_rename and 'OBS_VALUE' not in df.columns and 'TIME_PERIOD' not in df.columns:
             nums = df.select_dtypes(include=['number']).columns
             if len(nums) > 0:
                 mrv_cols = [c for c in nums if str(c).startswith('mrv_')]
                 if mrv_cols:
                     df = df.rename(columns={mrv_cols[-1]: 'OBS_VALUE'})
                     if val_label == "Value": val_label = str(mrv_cols[-1])
                 else:
                     target_col = nums[-1]
                     df = df.rename(columns={target_col: 'OBS_VALUE'})
                     if val_label == "Value": val_label = str(target_col)
        return df, val_label

    def _ensure_tidy(self, df):
        """Converts Wide (Years as columns) to Long/Tidy format if needed."""
        if 'TIME_PERIOD' in df.columns:
            return df
            
        # Detect Year Columns
        def _is_year(c):
            s = str(c)
            if s.isdigit(): return True
            try: 
                f = float(s)
                return f.is_integer() and 1900 < f < 2100
            except ValueError: return False
            
        year_cols = [c for c in df.columns if _is_year(c)]
        
        if year_cols:
            # Check for ID vars (everything else)
            id_vars = [c for c in df.columns if c not in year_cols]
            df_long = df.melt(id_vars=id_vars, value_vars=year_cols, 
                              var_name='TIME_PERIOD', value_name='OBS_VALUE')
            
            # Convert types
            df_long['TIME_PERIOD'] = pd.to_numeric(df_long['TIME_PERIOD'], errors='coerce')
            df_long['OBS_VALUE'] = pd.to_numeric(df_long['OBS_VALUE'], errors='coerce')
            return df_long
            
        return df

    def plot_trend(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Líneas con Jerarquía Visual (Protagonista vs Secundarias)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        
        # --- AUTO-DETECT INDEX AS TIME ---
        # If the index is named "Year", "Date", etc., or if it looks formatting, reset it to make it a column
        if df.index.name and df.index.name.lower() in ['year', 'time', 'date', 'period']:
             df = df.reset_index()
        elif pd.api.types.is_numeric_dtype(df.index) and (df.index > 1900).all() and (df.index < 2100).all():
             # Unnamed index but looks like years
             df.index.name = 'Year'
             df = df.reset_index()

        df = self._ensure_tidy(df)
        
        # --- AUTO-DETECT WIDE FORMAT (Year as Column, Series as Cols) ---
        # If we don't have standard keys, but have a "Year" or "Time" column
        if 'TIME_PERIOD' not in df.columns and 'REF_AREA' not in df.columns:
            time_cols = [c for c in df.columns if c.lower() in ['year', 'time', 'date', 'period']]
            if time_cols:
                t_col = time_cols[0]
                # Assume all other numeric cols are Series
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                series_vars = [c for c in num_cols if c != t_col]
                
                if series_vars:
                    # MELT IT
                    df = df.melt(id_vars=[t_col], value_vars=series_vars, 
                                 var_name='INDICATOR', value_name='OBS_VALUE')
                    df = df.rename(columns={t_col: 'TIME_PERIOD'})
                    
        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        hue_col = 'REF_AREA'
        if 'REF_AREA' in df.columns and df['REF_AREA'].nunique() == 1 and 'INDICATOR' in df.columns:
            hue_col = 'INDICATOR'
        
        # Determine unique hues/series
        unique_hues = df[hue_col].unique() if hue_col in df.columns else []
        
        # --- SMART SCALING (UNIT INTELLIGENCE) ---
        from wbgapi360.core.auditor import DataAuditor
        # We need the original dataframe or column-wise info? 
        # df here is melted (long). We need metadata about variables.
        # But scale conflict detection needs wide format or we check unique values per hue.
        # Let's check max value of each hue to classify.
        
        scale_map = {} # hue -> 'micro' | 'macro'
        for hue in unique_hues:
            max_val = df[df[hue_col] == hue]['OBS_VALUE'].abs().max()
            if max_val < 500:
                scale_map[hue] = 'micro'
            elif max_val > 1_000_000:
                scale_map[hue] = 'macro'
            else:
                scale_map[hue] = 'normal'
                
        has_micro = any(v == 'micro' for v in scale_map.values())
        has_macro = any(v == 'macro' for v in scale_map.values())
        
        use_dual_axis = has_micro and has_macro
        
        ax2 = None
        if use_dual_axis:
            print("[VISUALIZER] Smart Scaling Activated: DUAL AXIS Mode.")
            ax2 = ax.twinx()
            ax2.grid(False) # Only main grid

        # Manual Plotting for control over Linewidth/Alpha/Color
        for i, hue_val in enumerate(unique_hues):
            series_df = df[df[hue_col] == hue_val].dropna(subset=['OBS_VALUE', 'TIME_PERIOD']).sort_values('TIME_PERIOD')
            if series_df.empty: continue
            
            # Determine Axis
            scale_type = scale_map.get(hue_val, 'normal')
            use_ax2 = (scale_type == 'micro' and use_dual_axis)
            target_ax = ax2 if use_ax2 else ax
            
            # Style Logic
            # Cycle colors
            color = FT_PALETTE[i % len(FT_PALETTE)]
            
            linestyle = '--' if use_ax2 else '-'
            linewidth = 2.5
            
            line, = target_ax.plot(
                series_df['TIME_PERIOD'], 
                series_df['OBS_VALUE'], 
                color=color, 
                linewidth=linewidth, 
                linestyle=linestyle,
                marker='o' if use_ax2 else '',
                markersize=4,
                alpha=0.9, 
                label=str(hue_val)
            )
            
            # Label
            last_pt = series_df.iloc[-1]
            target_ax.text(
                x=last_pt['TIME_PERIOD'], 
                y=last_pt['OBS_VALUE'], 
                s=f"  {hue_val}", 
                color=color, 
                va='center', 
                ha='left', 
                fontsize=10, 
                weight='bold'
            )
 
        # Labels & Axis
        # Note: Title/subtitle handled by _finalize_chart to avoid duplication
        if ax2:
            ax.set_ylabel("Macro Scale (e.g. Billions)", color=FT_AXIS_COLOR)
            ax2.set_ylabel("Micro Scale (e.g. %)", color=FT_AXIS_COLOR)
            
        sns.despine(ax=ax, left=True, bottom=False)
        if ax2:
            sns.despine(ax=ax2, left=True, bottom=False, right=False)
            
        return self._finalize_chart(fig, ax, title, subtitle, save_path=save_path)

    def plot_bar(self, df, title="", subtitle="", save_path=None, figsize=None):
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        actual_figsize = figsize or (8, 6)
        self._current_figsize = actual_figsize  # Store for font scaling
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        df_sorted = df.sort_values('OBS_VALUE', ascending=False)
        fonts = self._get_scaled_fonts()
        
        sns.barplot(
            data=df_sorted, x='OBS_VALUE', y='REF_AREA',
            color=FT_BLUE, ax=ax, edgecolor=None 
        )
        
        # Scale y-axis labels (country names)
        ax.tick_params(axis='y', labelsize=fonts['axis_label'])
        
        ax.xaxis.grid(True, color=FT_GRID_COLOR) 
        ax.yaxis.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        self._format_axis(ax, 'x')
        self._finalize_chart(fig, ax, title, subtitle, save_path)


    def plot_column(self, df, title="", subtitle="", save_path=None, figsize=None):
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        sns.barplot(
            data=df, x='REF_AREA', y='OBS_VALUE',
            color=FT_BLUE, ax=ax, edgecolor=None
        )
        
        ax.yaxis.grid(True, color=FT_GRID_COLOR)
        ax.xaxis.grid(False)
        ax.set_ylabel("")
        ax.set_xlabel("")
        self._format_axis(ax, 'y')
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_scatter(self, df, title="", subtitle="", save_path=None, figsize=None):
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols = [c for c in numeric_cols if c != 'TIME_PERIOD']
        
        if len(cols) >= 2:
            x_col, y_col = cols[0], cols[1]
            sns.scatterplot(
                data=df, x=x_col, y=y_col, hue='REF_AREA' if 'REF_AREA' in df.columns else None,
                s=120, palette=FT_PALETTE, alpha=0.9, edgecolor='white', ax=ax, legend=False
            )
            ax.yaxis.grid(True, color=FT_GRID_COLOR)
            ax.xaxis.grid(True, color=FT_GRID_COLOR)
            ax.set_xlabel(x_col, color=FT_AXIS_COLOR)
            ax.set_ylabel(y_col, color=FT_AXIS_COLOR)
            self._format_axis(ax, 'both')
        else:
            ax.text(0.5, 0.5, "Insufficient numeric data", ha='center', color=FT_AXIS_COLOR)

        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_dumbbell(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Dumbbell (Comparación entre dos puntos)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        df = self._ensure_tidy(df)
        
        # Logic: We need exactly 2 time points or we take Min/Max time
        if 'TIME_PERIOD' not in df.columns:
            print("⚠️ Dumbbell plot requires 'TIME_PERIOD'.")
            return

        min_t, max_t = df['TIME_PERIOD'].min(), df['TIME_PERIOD'].max()
        if min_t == max_t:
            print("⚠️ Dumbbell plot requires distinct time periods.")
            return
            
        # Filter for only start and end
        df_dumb = df[df['TIME_PERIOD'].isin([min_t, max_t])].copy()
        
        # Pivot to get Start and End as columns
        # Index: REF_AREA, Columns: TIME_PERIOD, Values: OBS_VALUE
        try:
            df_wide = df_dumb.pivot(index='REF_AREA', columns='TIME_PERIOD', values='OBS_VALUE')
        except ValueError:
             # Handle duplicates if needed
             df_wide = df_dumb.groupby(['REF_AREA', 'TIME_PERIOD'])['OBS_VALUE'].mean().unstack()

        df_wide = df_wide.dropna()
        df_wide = df_wide.sort_values(by=max_t, ascending=True) # Sort by final value

        fig, ax = plt.subplots(figsize=(10, len(df_wide)*0.5 + 2))
        
        # Draw Lines (The "Bar")
        ax.hlines(y=df_wide.index, xmin=df_wide[min_t], xmax=df_wide[max_t], 
                  color=FT_NEUTRAL_LINE, alpha=0.5, linewidth=1.5)
        
        # Draw Points - Make it exciting!
        # Start: Red (Past/Warning?) or Neutral? User said "boring".
        # Let's do Start=Muted Red, End=Petrol Green (Transformation concept)
        ax.scatter(df_wide[min_t], df_wide.index, color=FT_RED, s=100, label=str(min_t), zorder=3)
        # End: Petrol Green (Dominant)
        ax.scatter(df_wide[max_t], df_wide.index, color=FT_GREEN, s=120, label=str(max_t), zorder=4)
        
        # Annotate difference? Optional. Let's keep it clean.
        
        # Legend
        ax.legend(title="", frameon=False, loc='upper right', bbox_to_anchor=(1, 1.1), ncol=2)
        
        ax.xaxis.grid(True, color=FT_GRID_COLOR)
        ax.yaxis.grid(False) # Categorical Y
        ax.set_ylabel("")
        ax.set_xlabel(val_label)
        self._format_axis(ax, 'x')
        
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_stacked(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Área/Barra Apilada (Composición)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        df = self._ensure_tidy(df)
        
        if 'TIME_PERIOD' not in df.columns or 'REF_AREA' not in df.columns:
            return

        # Prepare Wide Data
        df_wide = df.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE', aggfunc='sum')
        df_wide = df_wide.fillna(0)
        
        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        # Clean Palette for stack
        # FT uses muted colors for composition.
        colors = [FT_BLUE, FT_GREEN, FT_MUSTARD, FT_RED, FT_PURPLE, FT_NEUTRAL_DARK, FT_NEUTRAL_LIGHT]
        
        ax.stackplot(df_wide.index, df_wide.T, labels=df_wide.columns, colors=colors, alpha=0.9)
        
        # Legend (Necessary for composition)
        # FT Style: Legend above chart or direct label. Stacked area direct label is hard.
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)
        
        ax.yaxis.grid(True, color=FT_GRID_COLOR)
        ax.xaxis.grid(False)
        ax.set_ylabel(val_label)
        ax.set_xlabel("")
        ax.set_xlim(df_wide.index.min(), df_wide.index.max())
        self._format_axis(ax, 'y')
        
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_map(self, df, iso_col='REF_AREA', value_col='OBS_VALUE', title="", subtitle="", region=None, bbox=None, save_path=None, figsize=None):
        try:
            import geopandas as gpd
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                
                # GeoPandas 1.0+ deprecated datasets.get_path()
                # Use direct URL as fallback
                try:
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                except (AttributeError, ValueError):
                    # Fallback for GeoPandas >= 1.0
                    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                    world = gpd.read_file(url)
        except Exception as e:
            print(f"⚠️ Map generation skipped: GeoPandas/shapefile unavailable ({e})")
            return None
        
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        # Filter world geometries for regional centering
        region_bounds = None
        if region and region in REGION_BBOX:
            region_bounds = REGION_BBOX[region]
        elif bbox:
            region_bounds = bbox
        
        if region_bounds:
            from shapely.geometry import box
            region_box = box(region_bounds[0], region_bounds[1], region_bounds[2], region_bounds[3])
            world = world[world.geometry.intersects(region_box)]
        
        # Filter out unwanted territories
        try: 
            world = world[(world['POP_EST'] > 0)]
            world = world[world['NAME'] != "Antarctica"]
        except: 
            pass
        
        if 'TIME_PERIOD' in df.columns:
            latest_year = df['TIME_PERIOD'].max()
            df_map = df[df['TIME_PERIOD'] == latest_year].copy()
            subtitle = f"{subtitle} ({latest_year})" if subtitle else f"({latest_year})"
        else:
            df_map = df.copy()
        
        # --- LABELS=TRUE COMPATIBILITY ---
        # If REF_AREA_CODE exists, it means labels=True was used
        # Use the preserved ISO codes for GeoPandas matching
        if 'REF_AREA_CODE' in df_map.columns:
            # User requested human-readable labels, but we need codes for matching
            merge_col = 'REF_AREA_CODE'
        else:
            # Normal flow: use REF_AREA directly (it contains codes)
            merge_col = iso_col
        
        # Determine correct ISO column in shapefile (old vs new Natural Earth)
        iso_world_col = 'ISO_A3' if 'ISO_A3' in world.columns else 'iso_a3'
        
        world_data = world.merge(df_map, left_on=iso_world_col, right_on=merge_col, how='left')

        actual_figsize = figsize or DEFAULT_FIGSIZE_MAP
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        # Base world map (light neutral background)
        world.plot(ax=ax, color='#F6F6F6', edgecolor='#E0E0E0', linewidth=0.3)
        
        if not world_data.dropna(subset=[value_col]).empty:
            # FT-Style Purple Sequential Palette (Lavender → Deep Purple)
            # Modern, elegant, premium editorial look
            from matplotlib.colors import LinearSegmentedColormap
            from matplotlib.ticker import FuncFormatter
            
            # Purple palette: light to dark
            ft_map_colors = ['#F3E5F5', '#CE93D8', '#AB47BC', '#8E24AA', '#6A1B9A', '#4A148C']
            ft_cmap = LinearSegmentedColormap.from_list('ft_map', ft_map_colors, N=256)
            
            # Number formatter for legend
            def format_map_number(x, pos):
                if abs(x) >= 1e9:
                    return f'{x/1e9:.1f}B'
                elif abs(x) >= 1e6:
                    return f'{x/1e6:.1f}M'
                elif abs(x) >= 1e3:
                    return f'{x/1e3:,.0f}K'
                else:
                    return f'{x:,.0f}'
            
            # Clean label (remove mrv_, TIME_PERIOD suffixes)
            clean_label = val_label.replace('mrv_', '').replace('_', ' ').title() if val_label else ''
            
            world_data.dropna(subset=[value_col]).plot(
                column=value_col, 
                ax=ax, 
                legend=True,
                cmap=ft_cmap,
                edgecolor='#FFFFFF', 
                linewidth=0.4,
                legend_kwds={
                    'shrink': 0.4,  # Smaller legend
                    'label': clean_label,
                    'orientation': 'horizontal',
                    'pad': 0.02,
                    'format': FuncFormatter(format_map_number),
                    'aspect': 15  # Make legend thinner
                },
                missing_kwds={'color': '#F6F6F6'}
            )
        else:
            ax.text(0.5, 0.5, "No data available", 
                   ha='center', va='center', 
                   transform=ax.transAxes,
                   fontsize=14, color=FT_SUBTITLE_COLOR)
        
        
        
        ax.set_axis_off()
        is_regional = region is not None or bbox is not None
        self._finalize_chart(fig, ax, title, subtitle, save_path, is_regional_map=is_regional)

    def plot_map_bubble(self, df, iso_col='REF_AREA', value_col='OBS_VALUE', title="", subtitle="", region=None, bbox=None, save_path=None, figsize=None):
        """Bubble Map: Proportional circles over country centroids."""
        try:
            import geopandas as gpd
            import numpy as np
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                
                try:
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                except (AttributeError, ValueError):
                    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                    world = gpd.read_file(url)
        except Exception as e:
            print(f"⚠️ Map generation skipped: GeoPandas/shapefile unavailable ({e})")
            return None
        
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        try: 
            world = world[(world['POP_EST'] > 0)]
            world = world[world['NAME'] != "Antarctica"]
        except: 
            pass
        
        if 'TIME_PERIOD' in df.columns:
            latest_year = df['TIME_PERIOD'].max()
            df_map = df[df['TIME_PERIOD'] == latest_year].copy()
            subtitle = f"{subtitle} ({latest_year})" if subtitle else f"({latest_year})"
        else:
            df_map = df.copy()
        
        if 'REF_AREA_CODE' in df_map.columns:
            merge_col = 'REF_AREA_CODE'
        else:
            merge_col = iso_col
        
        iso_world_col = 'ISO_A3' if 'ISO_A3' in world.columns else 'iso_a3'
        world_data = world.merge(df_map, left_on=iso_world_col, right_on=merge_col, how='inner')
        
        actual_figsize = figsize or DEFAULT_FIGSIZE_MAP
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        world.plot(ax=ax, color='#F6F6F6', edgecolor='#E0E0E0', linewidth=0.3)
        
        if not world_data.empty:
            world_data['centroid'] = world_data.geometry.centroid
            max_val = world_data[value_col].max()
            min_val = world_data[value_col].min()
            world_data['bubble_size'] = ((world_data[value_col] - min_val) / (max_val - min_val)) * 1800 + 200
            
            ax.scatter(
                world_data.centroid.x, 
                world_data.centroid.y,
                s=world_data['bubble_size'],
                c='#AB47BC',
                alpha=0.6,
                edgecolor='#4A148C',
                linewidth=1.2
            )
        
        ax.set_axis_off()
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_heatmap(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Mapa de Calor (Correlación/Densidad)."""
        self._apply_theme_context()
        df = self._ensure_tidy(df)

    def plot_map_diverging(self, df, iso_col='REF_AREA', value_col='OBS_VALUE', center=None, title="", subtitle="", region=None, bbox=None, save_path=None, figsize=None):
        """Diverging Map: Red-White-Blue for positive/negative values."""
        try:
            import geopandas as gpd
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                
                try:
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                except (AttributeError, ValueError):
                    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                    world = gpd.read_file(url)
        except Exception as e:
            print(f"⚠️ Map generation skipped: GeoPandas/shapefile unavailable ({e})")
            return None
        
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        try: 
            world = world[(world['POP_EST'] > 0)]
            world = world[world['NAME'] != "Antarctica"]
        except: 
            pass
        
        if 'TIME_PERIOD' in df.columns:
            latest_year = df['TIME_PERIOD'].max()
            df_map = df[df['TIME_PERIOD'] == latest_year].copy()
            subtitle = f"{subtitle} ({latest_year})" if subtitle else f"({latest_year})"
        else:
            df_map = df.copy()
        
        if 'REF_AREA_CODE' in df_map.columns:
            merge_col = 'REF_AREA_CODE'
        else:
            merge_col = iso_col
        
        iso_world_col = 'ISO_A3' if 'ISO_A3' in world.columns else 'iso_a3'
        world_data = world.merge(df_map, left_on=iso_world_col, right_on=merge_col, how='left')
        
        actual_figsize = figsize or DEFAULT_FIGSIZE_MAP
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        world.plot(ax=ax, color='#F6F6F6', edgecolor='#E0E0E0', linewidth=0.3)
        
        if not world_data.dropna(subset=[value_col]).empty:
            from matplotlib.colors import TwoSlopeNorm
            from matplotlib.ticker import FuncFormatter
            
            if center is None:
                center = 0 if df_map[value_col].min() < 0 else df_map[value_col].median()
            
            divnorm = TwoSlopeNorm(vmin=df_map[value_col].min(), 
                                   vcenter=center, 
                                   vmax=df_map[value_col].max())
            
            def format_div(x, pos):
                if abs(x) >= 1e9:
                    return f'{x/1e9:.1f}B'
                elif abs(x) >= 1e6:
                    return f'{x/1e6:.1f}M'
                elif abs(x) >= 1e3:
                    return f'{x/1e3:,.0f}K'
                else:
                    return f'{x:,.1f}'
            
            world_data.plot(
                column=value_col,
                ax=ax,
                cmap='RdBu_r',
                norm=divnorm,
                legend=True,
                edgecolor='#FFFFFF',
                linewidth=0.4,
                legend_kwds={
                    'shrink': 0.6,
                    'label': val_label,
                    'orientation': 'horizontal',
                    'pad': 0.05,
                    'format': FuncFormatter(format_div)
                },
                missing_kwds={'color': '#F6F6F6'}
            )
        
        ax.set_axis_off()
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_map_categorical(self, df, iso_col='REF_AREA', value_col='OBS_VALUE', bins=None, labels=None, title="", subtitle="", region=None, bbox=None, save_path=None, figsize=None):
        """Categorical Map: Discrete color categories for classifications."""
        try:
            import geopandas as gpd
            import numpy as np
            import pandas as pd
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                
                try:
                    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
                except (AttributeError, ValueError):
                    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                    world = gpd.read_file(url)
        except Exception as e:
            print(f"⚠️ Map generation skipped: GeoPandas/shapefile unavailable ({e})")
            return None
        
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        try: 
            world = world[(world['POP_EST'] > 0)]
            world = world[world['NAME'] != "Antarctica"]
        except: 
            pass
        
        if 'TIME_PERIOD' in df.columns:
            latest_year = df['TIME_PERIOD'].max()
            df_map = df[df['TIME_PERIOD'] == latest_year].copy()
            subtitle = f"{subtitle} ({latest_year})" if subtitle else f"({latest_year})"
        else:
            df_map = df.copy()
        
        if 'REF_AREA_CODE' in df_map.columns:
            merge_col = 'REF_AREA_CODE'
        else:
            merge_col = iso_col
        
        iso_world_col = 'ISO_A3' if 'ISO_A3' in world.columns else 'iso_a3'
        world_data = world.merge(df_map, left_on=iso_world_col, right_on=merge_col, how='left')
        
        actual_figsize = figsize or DEFAULT_FIGSIZE_MAP
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        world.plot(ax=ax, color='#F6F6F6', edgecolor='#E0E0E0', linewidth=0.3)
        
        if not world_data.dropna(subset=[value_col]).empty:
            # Createte categories
            if bins is None:
                values = world_data[value_col].dropna()
                bins = [values.min(), values.quantile(0.25), values.quantile(0.5), 
                       values.quantile(0.75), values.max()]
            
            if labels is None:
                labels = [f"Q{i+1}" for i in range(len(bins)-1)]
            
            # Categorize values
            world_data['category'] = pd.cut(world_data[value_col], bins=bins, labels=labels, include_lowest=True)
            
            # Purple categorical palette (discrete colors)
            cat_colors = ['#F3E5F5', '#CE93D8', '#AB47BC', '#8E24AA', '#6A1B9A', '#4A148C']
            n_categories = len(labels)
            
            # Createte color map for categories
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(cat_colors[:n_categories])
            
            # Map category labels to numeric values for plotting
            world_data['category_num'] = world_data['category'].cat.codes
            
            # Plot with categorical colormap
            world_data.plot(
                column='category_num',
                ax=ax,
                cmap=cmap,
                edgecolor='#FFFFFF',
                linewidth=0.4,
                legend=False,
                categorical=True,
                missing_kwds={'color': '#F6F6F6'}
            )
            
            # Createte manual legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=cat_colors[i], label=labels[i]) 
                             for i in range(n_categories)]
            ax.legend(handles=legend_elements, loc='lower left', frameon=False, fontsize=10)
        
        ax.set_axis_off()
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_heatmap(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Mapa de Calor (Correlación/Densidad)."""
        self._apply_theme_context()
        df = self._ensure_tidy(df)
        
        # Standardize Matrix
        if 'TIME_PERIOD' in df.columns and 'REF_AREA' in df.columns and 'OBS_VALUE' in df.columns:
            df_wide = df.pivot_table(index='REF_AREA', columns='TIME_PERIOD', values='OBS_VALUE')
            
            # Sort by mean for hierarchy
            if not df_wide.empty:
                df_wide['mean'] = df_wide.mean(axis=1)
                df_wide = df_wide.sort_values('mean', ascending=False).drop(columns=['mean'])
        else:
            df_wide = df.select_dtypes(include=['number'])

        # Dynamic Height/Width based on data shape to avoid squashing
        h_ratios = len(df_wide) * 0.5 + 2
        w_ratios = len(df_wide.columns) * 0.8 + 2
        # Cap limits
        h_ratios = min(max(h_ratios, 6), 20)
        w_ratios = min(max(w_ratios, 8), 24)
        
        fig, ax = plt.subplots(figsize=(w_ratios, h_ratios))
        
        # Plot
        # Use simple fmt if values are large?
        sns.heatmap(df_wide, cmap="OrRd", annot=True, fmt=".1f", 
                    linewidths=0.5, linecolor='white', cbar_kws={'label': ''}, ax=ax)
        
        ax.set_ylabel("")
        ax.set_xlabel("") 
        plt.yticks(rotation=0)
        
        # Rotate X labels if there are many columns
        if len(df_wide.columns) > 5:
            plt.xticks(rotation=45, ha='right')
            
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_stacked_bar(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Columnas Apiladas (Composición)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        df = self._ensure_tidy(df)
        
        if 'TIME_PERIOD' not in df.columns or 'REF_AREA' not in df.columns:
            print("⚠️ Stacked Bar requires TIME_PERIOD and REF_AREA")
            return
            
        # Pivot for stacking
        try:
            df_pivot = df.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
        except ValueError:
            # Handle duplicates
            df_pivot = df.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE', aggfunc='mean')

        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        df_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=FT_PALETTE, edgecolor='white', linewidth=0.5)
        
        # Legend: Top Left, 1 row
        ax.legend(title="", bbox_to_anchor=(0, 1.1), loc='upper left', ncol=min(len(df_pivot.columns), 4), frameon=False)
        ax.yaxis.grid(True, color=FT_GRID_COLOR)
        ax.xaxis.grid(False)
        ax.set_xlabel("") # Remove TIME_PERIOD label
        plt.xticks(rotation=0)
        
        # Annotate totals or segments? Maybe just keep it clean.
        
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_area(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Área Apilada (Evolución de Composición)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        df = self._ensure_tidy(df)
        
        if 'TIME_PERIOD' not in df.columns or 'REF_AREA' not in df.columns:
            return

        try:
            df_pivot = df.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
        except ValueError:
            df_pivot = df.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE', aggfunc='mean')

        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        df_pivot.plot(kind='area', stacked=True, ax=ax, alpha=0.9, color=FT_PALETTE, linewidth=0)
        
        # Legend: Top Left
        ax.legend(title="", bbox_to_anchor=(0, 1.1), loc='upper left', ncol=min(len(df_pivot.columns), 4), frameon=False)
        ax.yaxis.grid(True, color=FT_GRID_COLOR)
        ax.xaxis.grid(False)
        ax.set_xlabel("") # Remove TIME_PERIOD label
        ax.set_xlim(df_pivot.index.min(), df_pivot.index.max())
        
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_bump(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Bump (Ranking Changes)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df)
        df = self._ensure_tidy(df)
        
        if 'TIME_PERIOD' not in df.columns: return

        # Calculate Ranks
        df['Rank'] = df.groupby('TIME_PERIOD')['OBS_VALUE'].rank(ascending=False, method='first')
        
        # Filter top N to avoid clutter
        top_n = 10
        latest_year = df['TIME_PERIOD'].max()
        # Fix: handle case where latest_year is NaN or empty
        if pd.isna(latest_year): return

        top_entities = df[df['TIME_PERIOD'] == latest_year].nsmallest(top_n, 'Rank')['REF_AREA'].unique()
        df_filtered = df[df['REF_AREA'].isin(top_entities)]

        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        # Invert Y axis for Rank 1 at top
        ax.invert_yaxis()
        
        # Plot
        sns.lineplot(data=df_filtered, x='TIME_PERIOD', y='Rank', hue='REF_AREA', 
                     palette=FT_PALETTE, linewidth=3, marker='o', ax=ax, legend=False)
        
        # Labeling
        for entity in top_entities:
            last = df_filtered[(df_filtered['REF_AREA'] == entity) & (df_filtered['TIME_PERIOD'] == latest_year)]
            if not last.empty:
                ax.text(last['TIME_PERIOD'].values[0] + 0.2, last['Rank'].values[0], 
                        f" {entity}", va='center', fontsize=10, weight='bold', color=FT_AXIS_COLOR)

        ax.yaxis.grid(False) 
        ax.xaxis.grid(True, color=FT_GRID_COLOR)
        ax.set_yticks(range(1, top_n + 1))
        ax.set_ylabel("Rank")
        ax.set_xlabel("")
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_donut(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico de Donas (Part-to-Whole)."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        # Aggregation
        if 'REF_AREA' in df.columns:
            data = df.groupby('REF_AREA')['OBS_VALUE'].sum().sort_values(ascending=False).head(8)
        else:
            return

        actual_figsize = figsize or (6, 6)
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        
        colors = [FT_BLUE, FT_GREEN, FT_MUSTARD, FT_RED, FT_PURPLE, FT_NEUTRAL_DARK, FT_NEUTRAL_LIGHT]
        
        # Donut Chart with Labels OUTSIDE, % INSIDE
        wedges, texts, autotexts = ax.pie(data, labels=data.index, autopct='%1.1f%%', 
                                          startangle=90, colors=colors, 
                                          wedgeprops={'width': 0.4, 'edgecolor': 'white'},
                                          pctdistance=0.80, # Valid inside distance
                                          labeldistance=1.1) # Valid outside distance
        
        # Text Styling
        plt.setp(texts, size=11, color='black', weight='medium') # Labels Outside Black
        plt.setp(autotexts, size=9, weight="bold", color="white") # % Inside White
        
        self._finalize_chart(fig, ax, title, subtitle, save_path)

    def plot_treemap(self, df, title="", subtitle="", save_path=None, figsize=None):
        """Gráfico Treemap (Jerarquía simple) - Dependency Free."""
        self._apply_theme_context()
        df, val_label = self._prepare_data(df, auto_rename=True)
        
        if 'REF_AREA' not in df.columns: return
        
        # Prepare data: sort descending
        data = df.copy()
        data = data.groupby('REF_AREA')['OBS_VALUE'].sum().sort_values(ascending=False)
        normed = data / data.sum()
        
        # Simple Tiling Algorithm (Slice-and-Dice aproximation or squarify-lite)
        # We will use a Row-Column layout mechanism
        rects = []
        x, y, w, h = 0.0, 0.0, 100.0, 100.0
        
        # Simple recursive split (Binary Tree)
        def recursive_split(nodes, x, y, w, h):
            if len(nodes) == 1:
                return [{'label': nodes.index[0], 'value': nodes.iloc[0], 'x': x, 'y': y, 'w': w, 'h': h}]
            
            if len(nodes) == 0: return []
            
            # Split in half by value mass
            cumsum = nodes.cumsum()
            total = nodes.sum()
            split_idx = (cumsum - total/2).abs().argmin()
            
            # Ensure at least one item in left group if possible, else 0
            # If split_idx is last element, pull back
            if split_idx == len(nodes) - 1 and len(nodes) > 1:
                split_idx -= 1
            
            left = nodes.iloc[:split_idx+1]
            right = nodes.iloc[split_idx+1:]
            
            left_sum = left.sum()
            right_sum = right.sum()
            total_sum = left_sum + right_sum
            
            if total_sum == 0: return []
            
            check_w = w > h
            
            res = []
            if check_w: # Split Vertically (change X)
                w_left = w * (left_sum / total_sum)
                res.extend(recursive_split(left, x, y, w_left, h))
                res.extend(recursive_split(right, x + w_left, y, w - w_left, h))
            else: # Split Horizontally (change Y)
                h_left = h * (left_sum / total_sum)
                res.extend(recursive_split(left, x, y, w, h_left))
                res.extend(recursive_split(right, x, y + h_left, w, h - h_left))
            return res

        rects = recursive_split(normed, 0, 0, 100, 100)
        
        actual_figsize = figsize or DEFAULT_FIGSIZE
        self._current_figsize = actual_figsize
        fig, ax = plt.subplots(figsize=actual_figsize)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_axis_off()
        
        import matplotlib.patches as patches
        
        for i, r in enumerate(rects):
            color = FT_PALETTE[i % len(FT_PALETTE)]
            rect = patches.Rectangle((r['x'], r['y']), r['w'], r['h'], 
                                   linewidth=1, edgecolor='white', facecolor=color)
            ax.add_patch(rect)
            
            # Label if big enough
            if r['w'] > 5 and r['h'] > 5:
                # Format Value using FT logic
                val = r['value'] * data.sum()
                if val >= 1e9: val_str = f'{val*1e-9:,.1f}bn'.replace('.0bn', 'bn')
                elif val >= 1e6: val_str = f'{val*1e-6:,.1f}m'.replace('.0m', 'm')
                elif val >= 1e3: val_str = f'{val:,.0f}'
                else: val_str = f'{val:,.0f}'
                
                ax.text(r['x'] + r['w']/2, r['y'] + r['h']/2, 
                        f"{r['label']}\n{val_str}", 
                        ha='center', va='center', color='white', fontsize=9, weight='bold')

        self._finalize_chart(fig, ax, title, subtitle, save_path)

viz = Visualizer()
