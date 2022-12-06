var lineChart
var lineChart2
var lineChart3

$("#ticker-selector-submit").click(function () {

    let startDate = new Date(),
        responseHolder = $("#response"),
        ticker = $("#ticker").val();

    responseHolder.html("");
    if (lineChart) {
        lineChart.destroy();
        lineChart2.destroy();
        lineChart3.destroy();
    }

    if (ticker === "select") {
        responseHolder.html("Please select a company...")
        return false
    }

    let data = {
        "ticker": ticker
    };

    $("#loading-overlay").show()

    let ajax = $.ajax({
        type: "GET",
        url: "/prediction",
        contentType: "application/json",
        data: data,
    });

    let endDate = new Date();
    let seconds = (endDate - startDate) / 1000
    if (seconds < 1) {
        let delay = 1000;
        setTimeout(function() {
            updateFrontend(ajax);
        }, delay);
    } else {
        updateFrontend(ajax)
    }
});

function updateFrontend(ajax){
    ajax.done(function(res){
        $("#response").html(res);

    values_first = [1297,
    1288,
    1285,
    1580,
    1491,
    2132,
    2092,
    1722,
    1869,
    1599,
    1137,
    1213,
    1221,
    1547,
    1412]

    values_burt = [
        1297,
        1295,
        1293,
        1398,
        1491,
        1592,
        2692,
        1792,
        1899,
        1599,
        1197,
        1293,
        1291,
        1587,
        1982
    ]
        values_first2 = [
    1137,
    1213,
    1221,
    1547,
    1412]


    values_burt2 = [
        1197,
        1293,
        1291,
        1587,
        1982
    ]

     labels2 =[
            "01-01-2020",
            "02-01-2020",
            "03-01-2020",
            "04-01-2020",
            "05-01-2020",
        ]

        labels =[
            "01-01-2020",
            "02-01-2020",
            "03-01-2020",
            "04-01-2020",
            "05-01-2020",
            "06-01-2020",
            "07-01-2020",
            "08-01-2020",
            "09-01-2020",
            "10-01-2020",
            "11-01-2020",
            "12-01-2020",
            "13-01-2020",
            "14-01-2020",
            "15-01-2020",
        ]

    let ctx = $("#lineChart")[0].getContext('2d');
    lineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Predicted",
                    data: values_first,
                    fill: false,
                    borderColor: "red",
                    lineTension: 0.1
                },
                {
                    label: "Actual",
                    data: values_burt,
                    fill: false,
                    borderColor: "blue",
                    lineTension: 0.1
                },
            ]
        },
        options: {
            responsive: false,
            maintainAspectRatio: true,
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: 'rgb(255, 99, 132)'
                    },
                    title: {
                        display: true,
                        text: "All Time Predictions",
                        padding: 10,
                        font: {
                            size: 20
                        }
                    }
                }
            }
        }
    });

    let ctx2 = $("#lineChart2")[0].getContext('2d');
    lineChart2 = new Chart(ctx2, {
        type: "line",
        data: {
            labels: labels2,
            datasets: [
                {
                    label: "Predicted",
                    data: values_first2,
                    fill: false,
                    borderColor: "red",
                    lineTension: 0.1
                },
                {
                    label: "Actual",
                    data: values_burt2,
                    fill: false,
                    borderColor: "blue",
                    lineTension: 0.1
                },
            ]
        },
        options: {
            responsive: false,
            maintainAspectRatio: true,
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: 'rgb(255, 99, 132)'
                    },
                    title: {
                        display: true,
                        text: "Past 5 Days",
                        padding: 10,
                        font: {
                            size: 20
                        }
                    }
                }
            }
        }
    });




     values_first3 = [
    1137,
    1213,
    1221,
    1547,
    1412,

    1422,
    1632,
    1532,
     ]


    values_burt3 = [
        1197,
        1293,
        1291,
        1587,
        1982
    ]

     labels2 =[
            "01-01-2020",
            "02-01-2020",
            "03-01-2020",
            "04-01-2020",
            "05-01-2020",
            "06-01-2020",
            "07-01-2020",
            "08-01-2020",
        ]



    let ctx3 = $("#lineChart3")[0].getContext('2d');
        lineChart3 = new Chart(ctx3, {
            type: "line",
            data: {
                labels: labels2,
                datasets: [
                    {
                        label: "Predicted",
                        data: values_first3,
                        fill: false,
                        borderColor: "red",
                        lineTension: 0.1
                    },
                    {
                        label: "Actual",
                        data: values_burt3,
                        fill: false,
                        borderColor: "blue",
                        lineTension: 0.1
                    },
                ]
            },
            options: {
                responsive: false,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45,
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: 'rgb(255, 99, 132)'
                        },
                        title: {
                            display: true,
                            text: "Future",
                            padding: 10,
                            font: {
                                size: 20
                            }
                        }
                    }
                }
            }
        });






        $("#loading-overlay").hide();
    });

    ajax.fail(function(res){
        $("#response").html(res.responseJSON.message)
         $("#loading-overlay").hide();
    });
}