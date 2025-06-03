import math
import pandas as pd
import streamlit as st
import altair as alt

# --- Functions ---
def square_law(x0, y0, a, b):
    def x_(t):
        ab_sqrt = math.sqrt(a * b)
        return x0 * math.cosh(ab_sqrt * t) - y0 * math.sqrt(b / a) * math.sinh(ab_sqrt * t)
    def y_(t):
        ab_sqrt = math.sqrt(a * b)
        return y0 * math.cosh(ab_sqrt * t) - x0 * math.sqrt(a / b) * math.sinh(ab_sqrt * t)
    t = 0.0
    x_vec, y_vec, time_vec = [], [], []
    x, y = x0, y0
    while x > 0 and y > 0:
        time_vec.append(t)
        x = x_(t)
        y = y_(t)
        x_vec.append(x)
        y_vec.append(y)
        t += 0.1
    return pd.DataFrame({'time': time_vec, 'x': x_vec, 'y': y_vec})

def parse_tags(tag_string):
    if pd.isna(tag_string):
        return []
    return [tag.strip() for tag in str(tag_string).split(',')]

def aggregate_force_advantaged(
    combat_power_df, my_units, opp_units, advantage_rules,
    strength_dict=None, opp_strength_dict=None, unit_effort_dict=None):

    subset = combat_power_df[combat_power_df['Unit'].isin(my_units)].copy()
    opp_subset = combat_power_df[combat_power_df['Unit'].isin(opp_units)].copy()

    if strength_dict:
        subset['Adj_Size'] = subset.apply(lambda row: row['Size'] * strength_dict.get(row['Unit'], 100) / 100.0, axis=1)
    else:
        subset['Adj_Size'] = subset['Size']
    if opp_strength_dict:
        opp_subset['Adj_Size'] = opp_subset.apply(lambda row: row['Size'] * opp_strength_dict.get(row['Unit'], 100) / 100.0, axis=1)
    else:
        opp_subset['Adj_Size'] = opp_subset['Size']

    total_size = subset['Adj_Size'].sum()
    if total_size == 0 or len(subset) == 0 or len(opp_subset) == 0:
        return 0, 0

    effective_cp_total = 0
    for _, my_row in subset.iterrows():
        my_cp = my_row['Combat Power']
        my_tags = my_row['Tags']
        my_size = my_row['Adj_Size']
        my_unit = my_row['Unit']
        my_effort = unit_effort_dict.get(my_unit, 1.0) if unit_effort_dict else 1.0

        opp_total_size = opp_subset['Adj_Size'].sum()
        if opp_total_size == 0:
            avg_multiplier = 1.0
        else:
            weighted_multipliers = []
            for _, opp_row in opp_subset.iterrows():
                opp_tags = opp_row['Tags']
                opp_size = opp_row['Adj_Size']
                pair_max = max([
                    advantage_rules.get((my_tag, opp_tag), 1.0)
                    for my_tag in my_tags for opp_tag in opp_tags
                ] or [1.0])
                weighted_multipliers.append(pair_max * opp_size)
            avg_multiplier = sum(weighted_multipliers) / opp_total_size

        effective_cp_total += my_cp * my_size * avg_multiplier * my_effort

    weighted_cp = effective_cp_total / total_size
    return total_size, weighted_cp

def adjudicate_day(
    combat_power_df, blue_selected, red_selected,
    blue_posture='Attack', prepared_defense=False,
    day_length=1.0, cp_scale=0.2, advantage_rules=None):

    # blue_selected: [(unit, pct, roe)], red_selected: [(unit, pct)]
    blue_units = [u for u, s, r in blue_selected]
    blue_strengths = {u: s for u, s, r in blue_selected}
    blue_effort = {u: r for u, s, r in blue_selected}
    red_units  = [u for u, s in red_selected]
    red_strengths  = {u: s for u, s in red_selected}

    blue_size, blue_cp = aggregate_force_advantaged(
        combat_power_df, blue_units, red_units, advantage_rules,
        strength_dict=blue_strengths, opp_strength_dict=red_strengths,
        unit_effort_dict=blue_effort
    )
    red_size, red_cp = aggregate_force_advantaged(
        combat_power_df, red_units, blue_units, advantage_rules,
        strength_dict=red_strengths, opp_strength_dict=blue_strengths
    )

    if blue_size == 0 or red_size == 0:
        return None

    blue_cp_multiplier = 1.0
    red_cp_multiplier = 1.0
    if blue_posture == 'Defend' and prepared_defense:
        blue_cp_multiplier = 1.33
    elif blue_posture == 'Attack' and prepared_defense:
        red_cp_multiplier = 1.33

    outcome = square_law(
        blue_size, red_size,
        blue_cp * blue_cp_multiplier * cp_scale,
        red_cp * red_cp_multiplier * cp_scale
    )
    if outcome is None:
        return None
    row = outcome.iloc[(outcome['time'] - day_length).abs().argmin()]
    blue_survivors = row['x']
    red_survivors = row['y']
    blue_pct = 100 * blue_survivors / blue_size if blue_size else 0
    red_pct = 100 * red_survivors / red_size if red_size else 0
    return {
        'blue_names': blue_units,
        'red_names': red_units,
        'blue_survivors': blue_survivors,
        'red_survivors': red_survivors,
        'blue_pct': blue_pct,
        'red_pct': red_pct
    }

def get_per_unit_survivors(selected_units, total_size, survivors):
    """Distribute survivors proportionally across units."""
    per_unit = []
    if total_size == 0:
        return [(unit_tuple[0], 0, 0) for unit_tuple in selected_units]
    for unit_tuple in selected_units:
        if len(unit_tuple) == 3:
            unit, pct, _ = unit_tuple
        else:
            unit, pct = unit_tuple
        unit_row = combat_power[combat_power['Unit'] == unit].iloc[0]
        effective_size = unit_row['Size'] * pct / 100.0
        remaining = effective_size * survivors / total_size
        percent_remaining = 100 * remaining / unit_row['Size'] if unit_row['Size'] else 0
        per_unit.append((unit, remaining, percent_remaining))
    return per_unit

# --- Multipliers / Rules ---
advantage_rules = {
    ('INF',   'INF'):   1.0,
    ('INF',   'ARMOR'): 1.0,
    ('INF',   'ARTY'):  1.0,
    ('INF',   'AA'):    1.0,
    ('INF',   'AT'):    1.0,
    ('INF',   'AIR'):   1.0,

    ('ARMOR', 'INF'):   1.0,  
    ('ARMOR', 'ARMOR'): 1.0,
    ('ARMOR', 'ARTY'):  1.0,
    ('ARMOR', 'AA'):    1.0,
    ('ARMOR', 'AT'):    0.7,
    ('ARMOR', 'AIR'):   0.5,

    ('ARTY',  'INF'):   1.0,   
    ('ARTY',  'ARMOR'): 1.0,
    ('ARTY',  'ARTY'):  1.0,
    ('ARTY',  'AA'):    1.0,
    ('ARTY',  'AT'):    1.0,
    ('ARTY',  'AIR'):   1.0,

    ('AA',    'INF'):   1.0,
    ('AA',    'ARMOR'): 1.0,
    ('AA',    'ARTY'):  1.0,
    ('AA',    'AA'):    1.0,
    ('AA',    'AT'):    1.0,
    ('AA',    'AIR'):   5.0,

    ('AT',    'INF'):   1.0,
    ('AT',    'ARMOR'): 2.0,
    ('AT',    'ARTY'):  1.0,
    ('AT',    'AA'):    1.0,
    ('AT',    'AT'):    1.0,
    ('AT',    'AIR'):   0.5,

    ('AIR',   'INF'):   1.0,
    ('AIR',   'ARMOR'): 1.0,
    ('AIR',   'ARTY'):  1.0,
    ('AIR',   'AA'):    0.2,
    ('AIR',   'AT'):    1.0,
    ('AIR',   'AIR'):   1.0,
}

# --- Load Data ---
combat_power = pd.read_csv('combat_power.csv')
combat_power['Unit'] = combat_power['Unit'].str.strip()
combat_power['Tags'] = combat_power['Tags'].apply(lambda s: [x.strip() for x in str(s).split(',')] if pd.notna(s) else [])

blue_unit_names = combat_power[~combat_power['Unit'].str.contains('Red_', case=False)]['Unit'].tolist()
red_unit_names  = combat_power[combat_power['Unit'].str.contains('Red_', case=False)]['Unit'].tolist()

# --- Sidebar setup ---
st.sidebar.header("Force Selection")

def per_side_unit_selection(unit_names, color, effort_per_unit=False):
    selected = []
    st.sidebar.subheader(f"{color} Units")
    for unit in unit_names:
        strength_key = f"{color.lower()}_strength_{unit}"
        check_key = f"{color.lower()}_check_{unit}"
        cols = st.sidebar.columns([2, 1])
        checked = cols[0].checkbox(unit, value=False, key=check_key)
        pct = cols[1].number_input(
            "", min_value=0.0, max_value=999.0, value=100.0, step=1.0, key=strength_key,
            label_visibility="collapsed", format="%.0f"
        )
        if checked:
            if effort_per_unit:
                effort_opt = st.sidebar.selectbox(
                    f"Effort: {unit}",
                    ['Low', 'Medium', 'High', 'Custom'],
                    key=f"{color.lower()}_effort_{unit}"
                )
                roe_dict = {'Low': 0.5, 'Medium': 1.0, 'High': 2.0}
                roe_val = st.sidebar.number_input(
                    f"Custom ROE: {unit}",
                    value=1.0,
                    key=f"{color.lower()}_roe_custom_{unit}"
                ) if effort_opt == 'Custom' else roe_dict[effort_opt]
                selected.append((unit, pct, roe_val))
            else:
                selected.append((unit, pct))
    return selected

blue_selected = per_side_unit_selection(blue_unit_names, "Blue", effort_per_unit=True)
red_selected = per_side_unit_selection(red_unit_names, "Red", effort_per_unit=False)

st.sidebar.markdown("---")
posture = st.sidebar.radio("Blue Posture", ['Attack', 'Defend'])
prepared_def = st.sidebar.checkbox("Prepared to Defend", value=False)

# --- Main Page ---
st.title("Combat Day Adjudicator")

if st.button("Adjudicate Day"):
    if not blue_selected or not red_selected:
        st.warning("Select at least one unit for each side.")
    else:
        result = adjudicate_day(
            combat_power,
            blue_selected,  # now includes per-unit ROE
            red_selected,
            blue_posture=posture,
            prepared_defense=prepared_def,
            advantage_rules=advantage_rules
        )
        if result is None:
            st.error("One or both sides have zero strength.")
        else:
            blue_total = sum([combat_power[combat_power['Unit'] == unit].iloc[0]['Size'] * pct/100.0 for unit, pct, _ in blue_selected])
            red_total = sum([combat_power[combat_power['Unit'] == unit].iloc[0]['Size'] * pct/100.0 for unit, pct in red_selected])

            blue_per_unit = get_per_unit_survivors(blue_selected, blue_total, result['blue_survivors'])
            red_per_unit = get_per_unit_survivors(red_selected, red_total, result['red_survivors'])

            st.success("--- Results after 1 day ---")
            st.write(f"**Blue ({', '.join(result['blue_names'])}):** {result['blue_survivors']:.1f} ({result['blue_pct']:.1f}% of committed strength)")
            st.write(f"**Red ({', '.join(result['red_names'])}):** {result['red_survivors']:.1f} ({result['red_pct']:.1f}% of committed strength)")

            st.write("### Blue survivors by unit")
            for unit, survivors, pct_remain in blue_per_unit:
                st.write(f"{unit}: {survivors:.1f} ({pct_remain:.1f}% of full strength)")

            st.write("### Red survivors by unit")
            for unit, survivors, pct_remain in red_per_unit:
                st.write(f"{unit}: {survivors:.1f} ({pct_remain:.1f}% of full strength)")

            # ---- Visualization: Combat Power Breakdown ----

            def combat_power_breakdown(
                combat_power_df, blue_selected, red_selected,
                blue_posture, prepared_defense,
                cp_scale, advantage_rules
            ):
                def side_stats(selected, opp_selected, side, is_blue):
                    if is_blue:
                        unit_names = [u for u, s, r in selected]
                        strengths = {u: s for u, s, r in selected}
                        efforts = {u: r for u, s, r in selected}
                    else:
                        unit_names = [u for u, s in selected]
                        strengths = {u: s for u, s in selected}
                        efforts = None  # red doesn't get per-unit effort for now
                    
                    # Properly handle opp_selected structure for both sides:
                    opp_names = [unit_tuple[0] for unit_tuple in opp_selected]
                    opp_strengths = {unit_tuple[0]: unit_tuple[1] for unit_tuple in opp_selected}


                    base_size = sum([combat_power_df[combat_power_df['Unit'] == u].iloc[0]['Size'] for u in unit_names])
                    degraded_size = sum([
                        combat_power_df[combat_power_df['Unit'] == u].iloc[0]['Size'] * strengths[u] / 100.0 for u in unit_names
                    ])
                    # For base_cp, ignore effort
                    base_cp = (sum([
                        combat_power_df[combat_power_df['Unit'] == u].iloc[0]['Combat Power'] *
                        combat_power_df[combat_power_df['Unit'] == u].iloc[0]['Size'] *
                        strengths[u] / 100.0 for u in unit_names
                    ]) / degraded_size) if degraded_size > 0 else 0
                    opp_units = opp_names
                    adv_size, adv_cp = aggregate_force_advantaged(
                        combat_power_df, unit_names, opp_units, advantage_rules,
                        strength_dict=strengths, opp_strength_dict=opp_strengths,
                        unit_effort_dict=efforts
                    )
                    advantage_multiplier = adv_cp / base_cp if base_cp else 1.0

                    if is_blue:
                        rate_of_effort = sum([efforts[u] * strengths[u] for u in unit_names]) / sum([strengths[u] for u in unit_names]) if strengths else 1.0
                    else:
                        rate_of_effort = 1.0
                    if is_blue and blue_posture == "Defend" and prepared_defense:
                        prep_def = 1.33
                    elif not is_blue and blue_posture == "Attack" and prepared_defense:
                        prep_def = 1.33
                    else:
                        prep_def = 1
                    scaling = cp_scale
                    final = adv_cp * prep_def * scaling * degraded_size
                    return {
                        "Unit Size": base_size,
                        "Strength-Adjusted Size": degraded_size,
                        "Base Combat Power": base_cp,
                        "Advantage Multiplier": advantage_multiplier,
                        "Advantaged Combat Power": adv_cp,
                        "Rate of Effort": rate_of_effort,
                        "Prepared Defense": prep_def,
                        "Scaling": scaling,
                        "Final Effective CP": final
                    }

                blue_stats = side_stats(blue_selected, red_selected, "Blue", is_blue=True)
                red_stats = side_stats(red_selected, blue_selected, "Red", is_blue=False)

                df = pd.DataFrame({
                    "Factor": [
                        "Unit Size",
                        "Strength-Adjusted Size",
                        "Base Combat Power",
                        "Advantage Multiplier",
                        "Advantaged Combat Power",
                        "Rate of Effort",
                        "Prepared Defense",
                        "Scaling",
                        "Final Effective CP"
                    ],
                    "Blue": [
                        blue_stats["Unit Size"],
                        blue_stats["Strength-Adjusted Size"],
                        blue_stats["Base Combat Power"],
                        blue_stats["Advantage Multiplier"],
                        blue_stats["Advantaged Combat Power"],
                        blue_stats["Rate of Effort"],
                        blue_stats["Prepared Defense"],
                        blue_stats["Scaling"],
                        blue_stats["Final Effective CP"]
                    ],
                    "Red": [
                        red_stats["Unit Size"],
                        red_stats["Strength-Adjusted Size"],
                        red_stats["Base Combat Power"],
                        red_stats["Advantage Multiplier"],
                        red_stats["Advantaged Combat Power"],
                        red_stats["Rate of Effort"],
                        red_stats["Prepared Defense"],
                        red_stats["Scaling"],
                        red_stats["Final Effective CP"]
                    ]
                })
                return df

            breakdown_df = combat_power_breakdown(
                combat_power, blue_selected, red_selected,
                blue_posture=posture, prepared_defense=prepared_def,
                cp_scale=0.2, advantage_rules=advantage_rules
            )

            st.write("### Combat Power Breakdown")
            st.dataframe(breakdown_df.set_index("Factor"), use_container_width=True)

            # Bar chart for final effective CP
            final_cp_df = breakdown_df[breakdown_df["Factor"] == "Final Effective CP"].melt(id_vars="Factor", var_name="Side", value_name="Final Effective CP")
            chart = alt.Chart(final_cp_df).mark_bar().encode(
                x=alt.X('Side:N', title="Side"),
                y=alt.Y('Final Effective CP:Q', title="Final Effective Combat Power"),
                color='Side:N'
            ).properties(width=400, height=300)
            st.altair_chart(chart, use_container_width=True)
