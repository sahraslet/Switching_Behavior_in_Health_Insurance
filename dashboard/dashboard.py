import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import json
import altair as alt  

st.set_page_config(page_title="Insurance Churn Dashboard", layout="wide")

st.title("📊 Predictive Dashboard for Insurance Churn")
st.markdown("Upload preprocessed CSV or Excel file:")

uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx'])

@st.cache_data
def load_geojson():
    with open("bundeslaender_modified.geo.json", "r", encoding="utf-8") as f:
        return json.load(f)

geojson_data = load_geojson()





def make_donut_distribution(data, value_col, label_col, color_list=None, chart_title=""):
    total = data[value_col].sum()
    data["percent"] = (data[value_col] / total * 100).round(1)

    if color_list is None:
        color_list = ['#29b5e8', '#155F7A', '#27AE60', '#F39C12', '#E74C3C']  # fallback colors

    base = alt.Chart(data).encode(
        theta=alt.Theta(field=value_col, type="quantitative"),
        color=alt.Color(field=label_col, type="nominal", scale=alt.Scale(range=color_list)),
        tooltip=[label_col, value_col, "percent"]
    )

    donut = base.mark_arc(innerRadius=50, cornerRadius=8).properties(width=300, height=300)

    text = base.mark_text(radius=120, fontSize=14).encode(
        text=alt.Text("percent:Q", format=".1f")
    )

    return (donut + text).properties(title=chart_title)



if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File loaded.")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    st.markdown("### Raw data preview")
    st.dataframe(df.head())

    if 'Q18' in df.columns:
        df = df.drop(columns=['Q18'])

    # --- Model selection starts here ---
    model_options = {
        "Logistic Regression": "logistic_model.pkl",
        "Linear Discriminant Analysis": "lda_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "Light GBM": "lgbm_model.pkl",
        "SVM": "svm_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Adaboost": "adaboost_model.pkl",
        # Add more models here as needed
    }

    selected_model_name = st.sidebar.selectbox("Select model for prediction:", list(model_options.keys()))

    model_path = model_options[selected_model_name]

    try:
        model = joblib.load(model_path)
        st.success(f"✅ Loaded model: {selected_model_name}")
    except Exception as e:
        st.error(f"❌ Failed to load model '{selected_model_name}': {e}")
        st.stop()
    # --- Model selection ends here ---

    try:
        df['Predicted Q18'] = model.predict(df)
        st.success("✅ Predictions done.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()




    st.sidebar.header("Filter Options")

    if 'alter' in df.columns:
        min_age = int(df['alter'].min())
        max_age = int(df['alter'].max())
        age_range = st.sidebar.slider("Select age range:", min_age, max_age, (min_age, max_age))
        df = df[(df['alter'] >= age_range[0]) & (df['alter'] <= age_range[1])]
        

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        options = df[col].unique()
        selected = st.sidebar.multiselect(f"Filter by {col}:", options, default=list(options))
        df = df[df[col].isin(selected)]

    classes = sorted(df['Predicted Q18'].unique())
    selected_classes = st.sidebar.multiselect("Select predicted classes (Q18):", classes, default=classes)
    df_filtered = df[df['Predicted Q18'].isin(selected_classes)]

    color_theme_list = [
        'blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds',
        'rainbow', 'turbo', 'viridis'
    ]
    selected_color_theme = st.sidebar.selectbox('Select a color theme', color_theme_list, index=0)

    st.markdown("## Key Metrics")
    st.metric("Total records after filtering", len(df_filtered))
    for c in selected_classes:
        st.write(f"Class {c}: {(df_filtered['Predicted Q18'] == c).sum()} records")

    st.markdown("## Distribution of Predicted Classes")
    fig = px.histogram(df_filtered, x='Predicted Q18', color='Predicted Q18',
                       labels={"Predicted Q18": "Predicted Class"},
                       title="Predicted Classes Histogram",
                       color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig, use_container_width=True)

    if 'alter' in df_filtered.columns:
        st.markdown("## Age Distribution by Predicted Class")

        # Define custom, stronger colors for each class (adjust colors as you like)
        custom_colors = ["#2daeee", "#f1f666", "#72e5ca", "#f44343", "#b880ec"]  
        # The list can have as many colors as classes you have

        fig_age = px.histogram(
            df_filtered,
            x='alter',
            color='Predicted Q18',
            nbins=30,
            title="Age Distribution by Predicted Class",
            labels={"alter": "Age"},
            color_discrete_sequence=custom_colors
        )
        st.plotly_chart(fig_age, use_container_width=True)


    if 'region' in df_filtered.columns:
        col1, col2 = st.columns(2)
        with col1:
        # Choropleth maps by Bundesland
        
            st.markdown("## Prediction Count by Region")
            count_df = df_filtered.groupby('region')['Predicted Q18'].count().reset_index()
            count_df.columns = ['region', 'count']
            fig_count = px.choropleth(
                count_df,
                geojson=geojson_data,
                locations='region',
                color='count',
                featureidkey="id",
                color_continuous_scale=selected_color_theme,
                title="Number of Predictions per Region"
            )
            fig_count.update_geos(fitbounds="locations", visible=False)
            fig_count.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_count, use_container_width=True)

        
        with col2:
            st.markdown("## Most Common Class by Region")
            mode_df = df_filtered.groupby('region')['Predicted Q18'].agg(lambda x: x.mode().iloc[0]).reset_index()
            mode_df.columns = ['region', 'majority_class']
            fig_mode = px.choropleth(
                mode_df,
                geojson=geojson_data,
                locations='region',
                color='majority_class',
                featureidkey="id",
                color_continuous_scale=selected_color_theme,
                title="Most Common Predicted Class per Region"
            )
            fig_mode.update_geos(fitbounds="locations", visible=False)
            fig_mode.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_mode, use_container_width=True)
            
        st.markdown("## Proportion of Selected Class by Region")
        target_class = st.selectbox("Select class to show proportion:", classes, key="class_prop")
        class_counts = df_filtered[df_filtered['Predicted Q18'] == target_class].groupby('region').size()
        total_counts = df_filtered.groupby('region').size()
        proportion_df = (class_counts / total_counts).reset_index().fillna(0)
        proportion_df.columns = ['region', 'proportion']

        fig_prop = px.choropleth(
            proportion_df,
            geojson=geojson_data,
            locations='region',
            color='proportion',
            featureidkey="id",
            color_continuous_scale=selected_color_theme,
            title=f"Proportion of Class {target_class} per Bundesland"
        )
        fig_prop.update_geos(fitbounds="locations", visible=False)
        fig_prop.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig_prop, use_container_width=True)

        st.markdown("## Predicted Class Distribution by Income Bracket")

        income_labels = {
            1: "Unter 40.000 €",
            2: "40.000 € bis 66.599 €",
            3: "66.600 € bis 92.999 €",
            4: "93.000 € bis 113.999 €",
            5: "114.000 € oder mehr",
            6: "Keine Angabe"
        }

        if 'einkommen_persoenlich' in df_filtered.columns:
            income_codes = sorted(df_filtered['einkommen_persoenlich'].dropna().unique())
            income_options = [income_labels.get(code, f"Code {code}") for code in income_codes]

            selected_label = st.selectbox("Select income bracket:", income_options, key="income_bracket")
            selected_code = [code for code, label in income_labels.items() if label == selected_label][0]

            df_income = df_filtered[df_filtered['einkommen_persoenlich'] == selected_code]
            class_counts = df_income['Predicted Q18'].value_counts().reset_index()
            class_counts.columns = ['Predicted Q18', 'Count']

            donut_chart = make_donut_distribution(
                class_counts,
                value_col='Count',
                label_col='Predicted Q18',
                chart_title=f"Class Distribution for Income Bracket: {selected_label}"
            ).properties(height=400, width=400)

            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No 'einkommen_persoenlich' column found in data for donut chart.")



        st.markdown("## Numeric Feature Correlation Heatmap")

        # Define default features including the target
        default_features = [
                "einkommen_persoenlich",
                "geschlecht",
                "verbundenheit_mit_krankenkasse",
                "schlechte_erfahrungen",
                "region",
                "Predicted Q18"
            ]

            # Get all numeric columns from the dataframe for selection options
        all_numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

            # Show multiselect with defaults pre-selected, user can choose any numeric feature
        selected_features = st.multiselect(
                "Select features to include in correlation heatmap:",
                options=all_numeric_cols,
                default=[f for f in default_features if f in all_numeric_cols]
            )

        if len(selected_features) > 1:
                corr = df_filtered[selected_features].corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap (User Selected Features)",
                    color_continuous_scale=selected_color_theme
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
                st.write("Select at least two numeric features for correlation heatmap.")


    gender_labels = {
        1: "male",
        2: "female"
    }

    if 'geschlecht' in df_filtered.columns:
        st.markdown("## Predicted Class Distribution by Gender")

        # Get unique gender codes in data
        gender_codes_present = sorted(df_filtered['geschlecht'].dropna().unique())

        # Create layout columns based on how many gender groups exist
        cols = st.columns(len(gender_codes_present))

        for idx, code in enumerate(gender_codes_present):
            label = gender_labels.get(code, f"Code {code}")
            df_gender = df_filtered[df_filtered['geschlecht'] == code]

            # Count predicted classes
            class_counts = df_gender['Predicted Q18'].value_counts().reset_index()
            class_counts.columns = ['Predicted Q18', 'Count']

            donut_chart = make_donut_distribution(
                class_counts,
                value_col='Count',
                label_col='Predicted Q18',
                chart_title=f"{label}",
              # pass this to your chart function
            )
            
            # Define your custom color palette (one color per class)
            custom_colors = ["#0b93f4", "#dd0868", "#32f332"]  # blue, orange, green

            # Inside your gender chart loop:
            donut_chart = make_donut_distribution(
                class_counts,
                value_col='Count',
                label_col='Predicted Q18',
                chart_title=f"{label}",
                color_list=custom_colors  # pass color list here
)

            with cols[idx]:
                st.altair_chart(donut_chart, use_container_width=True)
    else:
        st.info("No 'geschlecht' column found in data for donut chart.")

    
    experience_labels = {
    0: "no bad experiences",
    1: "bad experiences"
}

    if 'schlechte_erfahrungen' in df_filtered.columns:
        st.markdown("## Predicted Class Distribution by Experience")

        # Get only the codes that exist in the data
        experience_codes_present = sorted(df_filtered['schlechte_erfahrungen'].dropna().unique())

        # Create two columns to show charts side-by-side
        cols = st.columns(len(experience_codes_present))

        for idx, code in enumerate(experience_codes_present):
            label = experience_labels.get(code, f"Code {code}")
            df_experience = df_filtered[df_filtered['schlechte_erfahrungen'] == code]

            # Count class predictions
            class_counts = df_experience['Predicted Q18'].value_counts().reset_index()
            class_counts.columns = ['Predicted Q18', 'Count']

            custom_colors = ["#f1f666", "#72e5ca", "#b880ec"]  # blue, orange, green

            donut_chart = make_donut_distribution(
                class_counts,
                value_col='Count',
                label_col='Predicted Q18',
                chart_title=f"{label}",
                color_list=custom_colors
            )

            # Display in its respective column
            with cols[idx]:
                st.altair_chart(donut_chart, use_container_width=True)
    else:
        st.info("No 'schlechte_erfahrungen' column found in data for donut chart.")


    if 'verbundenheit_mit_krankenkasse' in df_filtered.columns:
        st.markdown("## Connectedness with Health Insurance by Predicted Class")

        # Get all available classes
        available_classes = sorted(df_filtered['Predicted Q18'].unique())

        # Let user select which classes to include
        selected_classes = st.multiselect("Select class(es) to display:", available_classes, default=available_classes)

        if selected_classes:
            # Let user choose which class should be on top
            top_class = st.selectbox("Select class to display on top:", selected_classes)

            # Reorder: move top_class to end (since area_chart stacks last on top)
            ordered_classes = [cls for cls in selected_classes if cls != top_class] + [top_class]

            # Group and pivot
            grouped = df_filtered[df_filtered['Predicted Q18'].isin(ordered_classes)] \
                .groupby(['verbundenheit_mit_krankenkasse', 'Predicted Q18']) \
                .size() \
                .reset_index(name='count')

            area_data = grouped.pivot(index='verbundenheit_mit_krankenkasse', columns='Predicted Q18', values='count')
            area_data = area_data.fillna(0).sort_index()

            # Reorder columns
            area_data = area_data[ordered_classes]

            st.area_chart(area_data)
        else:
            st.warning("Please select at least one class.")





    st.markdown("## Filtered Data Preview")
    st.dataframe(df_filtered.head(50))

    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data as CSV", data=csv_data, file_name="filtered_predictions.csv")

else:
    st.info("Please upload a preprocessed CSV or Excel file to get started.")
