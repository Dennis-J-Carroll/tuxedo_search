document.addEventListener('DOMContentLoaded', function() {
    const trainBtn = document.getElementById('train-btn');
    const datasetInput = document.getElementById('dataset');
    const epochsInput = document.getElementById('epochs');
    const lossChart = document.getElementById('loss-chart');
    const modelSummary = document.getElementById('model-summary');

    const lossChartCtx = lossChart.getContext('2d');
    const lossChartData = {
        labels: [],
        datasets: [{
            label: 'Training Loss',
            data: [],
            borderColor: 'rgb(75, 192, 192)',
        }]
    };
    const lossChartConfig = {
        type: 'line',
        data: lossChartData,
    };
    const chart = new Chart(lossChartCtx, lossChartConfig);

    trainBtn.addEventListener('click', async function() {
        if (!datasetInput.files[0]) {
            alert('Please select a dataset file.');
            return;
        }
        if (!datasetInput.files[0]) {
            alert('Please select a dataset file.');
            return;
        }

        const formData = new FormData();
        formData.append('dataset', datasetInput.files[0]);
        formData.append('epochs', epochsInput.value);

        trainBtn.disabled = true;

        const progressDiv = document.getElementById('progress');
        progressDiv.style.display = 'block';

        try {
            trainBtn.disabled = true;

            const progressDiv = document.getElementById('progress');
            progressDiv.style.display = 'block';

            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }

                const data = await response.json();

                lossChartData.labels = data.epochs;
                lossChartData.datasets[0].data = data.loss;
                chart.update();

                modelSummary.innerText = data.summary;
            } catch (error) {
                console.error('Error during model training:', error);
                alert('An error occurred during model training. Please try again.');
            } finally {
                progressDiv.style.display = 'none';
                trainBtn.disabled = false;
            }

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();

            lossChartData.labels = data.epochs;
            lossChartData.datasets[0].data = data.loss;
            chart.update();

            modelSummary.innerText = data.summary;
        } catch (error) {
            console.error('Error during model training:', error);
            alert('An error occurred during model training. Please try again.');
        } finally {
            progressDiv.style.display = 'none';
            trainBtn.disabled = false;
        }

        const data = await response.json();
        
        lossChartData.labels = data.epochs;
        lossChartData.datasets[0].data = data.loss;
        chart.update();

        modelSummary.innerText = data.summary;
    });
});
