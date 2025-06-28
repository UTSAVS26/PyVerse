from flask_wtf import FlaskForm
from wtforms import SelectField, FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from flask_wtf import FlaskForm
from wtforms import (SelectField, FloatField, SubmitField, 
                     DateField, FileField)
from wtforms.validators import DataRequired, NumberRange

class PredictionForm(FlaskForm):
    model_name = SelectField(
        'Select Trained Model',
        choices=[
            ('linear_regression', 'Linear Regression'),
            ('ridge_regression', 'Ridge Regression'),
            ('lasso_regression', 'Lasso Regression'),
            ('svr', 'Support Vector Regression')
        ],
        validators=[DataRequired()]
    )
    
    # Key features for prediction
    total_bitcoins = FloatField(
        'Total Bitcoins in Circulation',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    trade_volume = FloatField(
        'Trade Volume (USD)',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    blocks_size = FloatField(
        'Blockchain Size (MB)',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    avg_block_size = FloatField(
        'Average Block Size (KB)',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    transactions_per_block = FloatField(
        'Transactions per Block',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    difficulty = FloatField(
        'Mining Difficulty',
        validators=[DataRequired(), NumberRange(min=0)]
    )
    
    submit = SubmitField('Predict Price')

class ModelForm(FlaskForm):
    model_type = SelectField(
        'Model Type',
        choices=[
            ('linear', 'Linear Regression'),
            ('ridge', 'Ridge Regression'),
            ('lasso', 'Lasso Regression'),
            ('svr', 'Support Vector Regression')
        ],
        validators=[DataRequired()]
    )
    
    alpha = FloatField(
        'Alpha (Regularization Strength)',
        validators=[
            DataRequired(),
            NumberRange(min=0.01, max=10.0)
        ],
        default=1.0
    )
    
    submit = SubmitField('Train Model')