document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    let predictionChart = null;

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading, hide results
        loadingDiv.style.display = 'block';
        resultsDiv.style.display = 'none';
        
        // Get form data
        const formData = {
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            wind: parseFloat(document.getElementById('wind').value),
            rain: parseFloat(document.getElementById('rain').value)
        };
        
        // In a real implementation, this would be a fetch call to your Flask endpoint
        // For now, we'll simulate the API response after a short delay
        simulatePrediction(formData).then(data => {
            if (data.status === 'success') {
                updateUIWithPredictions(data.prediction, formData);
            }
        });
    });
    
    function simulatePrediction(formData) {
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({
                    status: 'success',
                    prediction: {
                        ridge: calculateDummyPrediction(formData, 'ridge'),
                        polynomial: calculateDummyPrediction(formData, 'poly')
                    }
                });
            }, 1500); // Simulate network delay
        });
    }
    
    function calculateDummyPrediction(formData, modelType) {
        // Simple formula to generate somewhat realistic dummy data
        let base = (formData.temperature * 0.5) + 
                  ((100 - formData.humidity) * 0.3) + 
                  (formData.wind * 0.2) - 
                  (formData.rain * 2);
        
        // Adjust slightly between models
        if (modelType === 'poly') {
            base = base * 1.1;
        }
        
        // Ensure reasonable bounds
        base = Math.max(0, base);
        base = Math.min(100, base);
        
        return base.toFixed(2);
    }
    
    function updateUIWithPredictions(prediction, formData) {
        // Update prediction values
        document.getElementById('ridgePrediction').textContent = prediction.ridge;
        document.getElementById('polyPrediction').textContent = prediction.polynomial;
        
        // Update chart
        updateChart(prediction.ridge, prediction.polynomial);
        
        // Update insights
        updateInsights(prediction.ridge, prediction.polynomial, formData);
        
        // Show results, hide loading
        loadingDiv.style.display = 'none';
        resultsDiv.style.display = 'block';
    }
    
    function updateChart(ridgeValue, polyValue) {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        
        // Destroy previous chart if exists
        if (predictionChart) {
            predictionChart.destroy();
        }
        
        predictionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Ridge Regression', 'Polynomial Regression'],
                datasets: [{
                    label: 'Predicted Burned Area (hectares)',
                    data: [ridgeValue, polyValue],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Hectares'
                        }
                    }
                }
            }
        });
    }
    
    function updateInsights(ridgeValue, polyValue, inputData) {
        const insightsDiv = document.getElementById('insights');
        const avgPrediction = ((parseFloat(ridgeValue) + parseFloat(polyValue)) / 2);
        const formattedAvg = avgPrediction.toFixed(2);
        
        let riskLevel = 'low';
        let riskClass = 'risk-low';
        if (avgPrediction > 30) {
            riskLevel = 'high';
            riskClass = 'risk-high';
        } else if (avgPrediction > 15) {
            riskLevel = 'medium';
            riskClass = 'risk-medium';
        }
        
        insightsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>Average Prediction</h5>
                        <p><span class="${riskClass}" style="font-size: 1.5rem;">${formattedAvg} ha</span></p>
                        <p>Risk Level: <span class="${riskClass}">${riskLevel.toUpperCase()}</span></p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="insight-card">
                        <h5>Input Summary</h5>
                        <ul class="list-unstyled">
                            <li>Temperature: ${inputData.temperature}°C</li>
                            <li>Humidity: ${inputData.humidity}%</li>
                            <li>Wind Speed: ${inputData.wind} km/h</li>
                            <li>Rain: ${inputData.rain} mm/m²</li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="insight-card mt-3">
                <h5>Recommendations</h5>
                ${getRecommendations(riskLevel, inputData)}
            </div>
        `;
    }
    
    function getRecommendations(riskLevel, inputData) {
        let recommendations = '';
        
        if (riskLevel === 'high') {
            recommendations = `
                <div class="alert alert-danger">
                    <strong>High Fire Risk Detected!</strong> Immediate action recommended:
                    <ul>
                        <li>Alert local fire authorities</li>
                        <li>Prepare evacuation plans</li>
                        <li>Activate emergency protocols</li>
                    </ul>
                </div>
            `;
        } else if (riskLevel === 'medium') {
            recommendations = `
                <div class="alert alert-warning">
                    <strong>Moderate Fire Risk Detected.</strong> Precautionary measures:
                    <ul>
                        <li>Monitor conditions closely</li>
                        <li>Prepare fire suppression resources</li>
                        <li>Restrict access to high-risk areas</li>
                    </ul>
                </div>
            `;
        } else {
            recommendations = `
                <div class="alert alert-success">
                    <strong>Low Fire Risk.</strong> Standard precautions:
                    <ul>
                        <li>Maintain regular monitoring</li>
                        <li>Ensure firebreaks are clear</li>
                        <li>Educate staff on fire safety</li>
                    </ul>
                </div>
            `;
        }
        
        // Additional recommendations based on input
        if (inputData.humidity < 30) {
            recommendations += `
                <div class="alert alert-info mt-2">
                    <strong>Low Humidity Alert:</strong> Current humidity levels (${inputData.humidity}%) significantly increase fire risk.
                    Consider moisture retention strategies if possible.
                </div>
            `;
        }
        
        if (inputData.wind > 20) {
            recommendations += `
                <div class="alert alert-info mt-2">
                    <strong>High Winds Alert:</strong> Wind speeds (${inputData.wind} km/h) can rapidly spread fires.
                    Be prepared for fast-changing conditions.
                </div>
            `;
        }
        
        return recommendations;
    }
});