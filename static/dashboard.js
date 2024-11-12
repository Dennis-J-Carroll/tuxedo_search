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
        const formData = new FormData();
        formData.append('dataset', datasetInput.files[0]);
        formData.append('epochs', epochsInput.value);

        const response = await fetch('/train', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        lossChartData.labels = data.epochs;
        lossChartData.datasets[0].data = data.loss;
        chart.update();

        modelSummary.innerText = data.summary;
    });
});
