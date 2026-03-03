from flask import Flask, request, render_template

from src.Airbnb.pipelines.Prediction_pipeline import CustomData
from src.Airbnb.pipelines.Prediction_pipeline import PredictPipeline


app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route('/', methods=['GET', "POST"])
def home_page():
    if request.method == 'POST':
        try:
            data = CustomData(
                property_type=request.form.get("property_type"),
                room_type=request.form.get("room_type"),
                amenities=(request.form.get("amenities")),
                accommodates=(request.form.get("accommodates")),
                bathrooms=(request.form.get("bathrooms")),
                bed_type=request.form.get("bed_type"),
                cancellation_policy=request.form.get("cancellation_policy"),
                cleaning_fee=(request.form.get("cleaning_fee")),
                city=request.form.get("city"),
                host_has_profile_pic=request.form.get("dp"),
                host_identity_verified=request.form.get("verify"),
                host_response_rate=request.form.get("hostresponse"),
                instant_bookable=request.form.get("instant_bookable"),
                latitude=(request.form.get("latitude")),
                longitude=(request.form.get("lonlongitudeg")),
                number_of_reviews=(request.form.get("review")),
                review_scores_rating=(request.form.get("overallreview")),
                bedrooms=(request.form.get("bedrooms")),
                beds=(request.form.get("beds"))
            )

            final_data = data.get_data_as_dataframe()
            # print(final_data)
            
            predict_pipeline = PredictPipeline()
            pred_price = predict_pipeline.predict(final_data)
            
            result = round(pred_price[0], 2)
            
            return render_template('index.html', result=result)
            
        except Exception as e:
            return f"An error occurred: {e}"
   
    else:
       return render_template('index.html')

