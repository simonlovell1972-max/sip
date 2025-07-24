
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai

st.set_page_config(page_title="Dynamic Survey Insight App", layout="wide")
st.title("📊 Dynamic Survey Insight App")

uploaded_file = st.file_uploader("Upload your survey CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.rename(columns={df.columns[0]: "Location"}, inplace=True)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Identify likely question columns (excluding metadata)
    possible_questions = [col for col in df.columns if col.startswith("Q")]
    open_ended_cols = [col for col in possible_questions if df[col].dropna().apply(lambda x: isinstance(x, str) and len(x.strip().split()) > 3).mean() > 0.5]

    st.subheader("📊 Auto-generated Charts")
    for col in possible_questions:
        if col not in open_ended_cols:
            st.markdown(f"**{col}**")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.subheader("🧠 GPT Insight Summary (Open-Ended Questions)")
    from openai import OpenAI  # NEW IMPORT ABOVE — don't forget to add it

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)

    for col in open_ended_cols:
        st.markdown(f"**{col}**")
        text = "\n".join(df[col].dropna().astype(str).tolist())
        prompt = f"Summarise the following customer comments and identify key themes:\n{text[:4000]}"
        try:
            with st.spinner(f"Generating GPT insight for {col}..."):
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.success("Insight generated:")
                st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {e}")
                    )
                    st.success("Insight generated:")
                    st.write(response["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(f"Error: {e}")
