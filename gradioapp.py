import gradio as gr
import joblib
import pandas as pd
import os

model = joblib.load("MultiLinear_regression_Price_model_predict (1).pkl")


def predict_price(area, bedrooms, floors):

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Floors": [floors]
    })

    prediction = model.predict(input_data)[0]

    return f"Predicted House Price: ₹{prediction:.2f} Lakhs"


demo = gr.Interface(
    fn=predict_price,

    inputs=[
        gr.Number(
            label="Enter House Area (sq.ft)",
            minimum=600,
            maximum=3000,
            value=1200
        ),

        gr.Number(
            label="Enter Number of Bedrooms",
            minimum=1,
            maximum=4,
            value=2
        ),

        gr.Number(
            label="Enter Number of Floors",
            minimum=0,
            maximum=10,
            value=2
        )
    ],

    outputs=gr.Textbox(label="Predicted Price"),

    title="🏠 House Price Prediction",

    description="Predict house price based on Area, Bedrooms and Floors."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

