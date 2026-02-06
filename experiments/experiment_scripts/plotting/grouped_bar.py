"""Grouped bar plot utilities."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def grouped_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    bar_col: str,
    group_order: list[str] | None = None,
    bar_order: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend_title: str | None = None,
    legend_labels: dict[str, str] | None = None,
    colors: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    figsize: tuple[float, float] = (10, 6),
    bar_width: float = 0.8,
    log_scale: bool = False,
    log_ticks: list[float] | None = None,
    show_values: bool = False,
    value_format: str = ".0f",
    error_col: str | None = None,
    show_ci: bool = True,
    std_col: str = "std",
    n_samples_col: str = "n_samples",
    ci_level: float = 0.95,
    highlight_groups: list[list[str]] | None = None,
    highlight_color: str = "#f0f0f0",
    highlight_padding: float = 0.4,
    side_text: str | None = None,
    side_text_width: float = 0.25,
    side_text_color: str = "#f5f5f5",
    side_text_fontsize: int = 10,
    ax: plt.Axes | None = None,
    rotation: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar plot from a DataFrame.

    Args:
        df: DataFrame containing the data
        value_col: Column name for the bar values (y-axis)
        group_col: Column name for the groups (x-axis groupings)
        bar_col: Column name for the bars within each group (different colored bars)
        group_order: Optional list specifying order of groups on x-axis
        bar_order: Optional list specifying order of bars within each group
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        legend_title: Title for the legend
        legend_labels: Dict mapping bar_col values to display labels
        colors: List of colors for bars (in bar_order order)
        color_map: Dict mapping bar_col values to colors (alternative to colors)
        figsize: Figure size as (width, height)
        bar_width: Total width allocated for each group's bars (0-1)
        log_scale: Whether to use log scale for y-axis
        log_ticks: Custom tick values for log scale (displayed as normal numbers)
        show_values: Whether to show value labels on bars
        value_format: Format string for value labels (e.g., ".0f", ".1f")
        error_col: Optional column name for pre-computed error bars (takes precedence)
        show_ci: Whether to show 95% CI error bars (default True, uses std_col and n_samples_col)
        std_col: Column name for standard deviation (default "std")
        n_samples_col: Column name for sample count (default "n_samples")
        ci_level: Confidence level for CI calculation (default 0.95)
        highlight_groups: List of group lists to highlight with background
            e.g., [["base"]] or [["base"], ["ft1", "ft2"]] for multiple regions
        highlight_color: Background color for highlighted groups (default light gray)
        highlight_padding: Horizontal padding around highlighted region (default 0.4)
        side_text: Text to display in a box to the right of the plot
        side_text_width: Width of side text area as fraction of figure (default 0.25)
        side_text_color: Background color for side text box (default light gray)
        side_text_fontsize: Font size for side text (default 10)
        ax: Optional existing axes to plot on (note: side_text requires ax=None)

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> df = pd.DataFrame({
        ...     'model': ['base', 'base', 'ft1', 'ft1'],
        ...     'prefix': ['short', 'long', 'short', 'long'],
        ...     'length': [100, 500, 80, 200]
        ... })
        >>> fig, ax = grouped_bar_plot(
        ...     df,
        ...     value_col='length',
        ...     group_col='model',
        ...     bar_col='prefix',
        ...     bar_order=['short', 'long'],
        ...     colors=['#1f77b4', '#ff7f0e']
        ... )
    """
    # Determine group and bar orders
    if group_order is None:
        group_order = df[group_col].unique().tolist()
    if bar_order is None:
        bar_order = df[bar_col].unique().tolist()

    n_groups = len(group_order)
    n_bars = len(bar_order)

    # Set up colors
    if color_map is not None:
        colors = [color_map[bar] for bar in bar_order]
    elif colors is None:
        # Default color palette
        default_colors = plt.cm.tab10.colors
        colors = [default_colors[i % len(default_colors)] for i in range(n_bars)]

    # Create figure if needed
    if ax is None:
        if side_text:
            # Create figure with main plot and side text area
            fig = plt.figure(figsize=figsize)
            # Layout parameters
            left_margin = 0.08
            right_margin = 0.02
            bottom_margin = 0.12
            top_margin = 0.1
            gap = 0.03  # Gap between plot and text box
            plot_height = 1 - bottom_margin - top_margin
            # Calculate widths
            available_width = 1 - left_margin - right_margin
            plot_width = available_width - side_text_width - gap
            text_x = left_margin + plot_width + gap
            # Main plot
            ax = fig.add_axes([left_margin, bottom_margin, plot_width, plot_height])
            # Side text area
            text_ax = fig.add_axes([text_x, bottom_margin, side_text_width, plot_height])
            text_ax.set_facecolor(side_text_color)
            text_ax.set_xticks([])
            text_ax.set_yticks([])
            for spine in text_ax.spines.values():
                spine.set_visible(False)
            # Add text with manual word wrapping
            import textwrap
            # Estimate chars per line based on font size and box width
            # Box width in points = side_text_width * figsize[0] * 72 (72 points per inch)
            # Monospace char width ≈ 0.6 * fontsize points
            # Apply 0.85 factor for padding/margins
            box_width_points = side_text_width * figsize[0] * 72
            char_width_points = side_text_fontsize * 0.6
            chars_per_line = int((box_width_points / char_width_points) * 0.85)
            wrapped_lines = []
            for line in side_text.split('\n'):
                if len(line) <= chars_per_line:
                    wrapped_lines.append(line)
                else:
                    wrapped_lines.extend(textwrap.wrap(line, width=chars_per_line))
            wrapped_text = '\n'.join(wrapped_lines)
            text_ax.text(
                0.05, 0.95, wrapped_text,
                transform=text_ax.transAxes,
                fontsize=side_text_fontsize,
                verticalalignment='top',
                horizontalalignment='left',
                family='monospace',
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Calculate bar positions
    x = np.arange(n_groups)
    width = bar_width / n_bars

    # Determine if we can show error bars
    # Priority: error_col > calculated CI from std/n_samples
    can_calc_ci = (
        show_ci
        and std_col in df.columns
        and n_samples_col in df.columns
        and error_col is None
    )
    use_error_col = error_col is not None and error_col in df.columns
    show_error_bars = use_error_col or can_calc_ci

    # Z-score for CI level (default 1.96 for 95%)
    from scipy import stats
    z_score = stats.norm.ppf(1 - (1 - ci_level) / 2)

    # Draw highlight backgrounds for specified groups
    if highlight_groups:
        from matplotlib.patches import Rectangle
        # Get y-axis limits (will be auto-scaled, so we use a large range)
        # We'll adjust this after plotting if needed
        for group_list in highlight_groups:
            # Find indices of groups in this highlight region
            indices = [group_order.index(g) for g in group_list if g in group_order]
            if indices:
                min_idx = min(indices)
                max_idx = max(indices)
                # Draw rectangle spanning these groups
                rect_x = min_idx - highlight_padding
                rect_width = (max_idx - min_idx + 1) + 2 * highlight_padding - 1
                # Use axis transform for y to span full height
                rect = Rectangle(
                    (rect_x, 0),
                    rect_width,
                    1,
                    transform=ax.get_xaxis_transform(),
                    facecolor=highlight_color,
                    edgecolor='none',
                    zorder=0,
                )
                ax.add_patch(rect)

    # Plot each bar group
    bars_list = []
    for i, bar_val in enumerate(bar_order):
        bar_data = []
        error_data = [] if show_error_bars else None

        for group_val in group_order:
            mask = (df[group_col] == group_val) & (df[bar_col] == bar_val)
            matched = df[mask]
            if len(matched) > 0:
                bar_data.append(matched[value_col].values[0])
                if show_error_bars:
                    if use_error_col:
                        error_data.append(matched[error_col].values[0])
                    elif can_calc_ci:
                        std = matched[std_col].values[0]
                        n = matched[n_samples_col].values[0]
                        ci = z_score * std / np.sqrt(n) if n > 0 else 0
                        error_data.append(ci)
            else:
                bar_data.append(0)
                if show_error_bars:
                    error_data.append(0)

        # Calculate position offset for this bar within each group
        offset = (i - (n_bars - 1) / 2) * width

        # Get label for legend
        label = bar_val
        if legend_labels and bar_val in legend_labels:
            label = legend_labels[bar_val]

        # Plot bars
        bars = ax.bar(
            x + offset,
            bar_data,
            width,
            label=label,
            color=colors[i],
            yerr=error_data if show_error_bars else None,
            capsize=3 if show_error_bars else 0,
        )
        bars_list.append(bars)

        # Add value labels if requested
        if show_values:
            for bar, val in zip(bars, bar_data):
                if val > 0:
                    height = bar.get_height()
                    ax.annotate(
                        f'{val:{value_format}}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                    )

    # Configure axes
    ax.set_xticks(x)
    if rotation is not None:
        ax.set_xticklabels(group_order, rotation=rotation)
    else:
        ax.set_xticklabels(group_order)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Configure log scale
    if log_scale:
        ax.set_yscale('log')
        if log_ticks:
            ax.yaxis.set_major_locator(ticker.FixedLocator(log_ticks))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
        else:
            # Default log ticks at nice values
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

    # Add legend
    ax.legend(title=legend_title)

    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Only use tight_layout if we're not using manual axes positioning
    if not side_text:
        plt.tight_layout()

    return fig, ax
