var lineChart
var lineChart2
var lineChart3
var blurbText
var ticker

var buy = $("#response h2.buy"),
    sell = $("#response h2.sell"),
    hold = $("#response h2.hold");

$("#ticker-selector-submit").click(function () {

    let startDate = new Date(),
        responseInner = $("#response-inner");

    ticker = $("#ticker").val();
    blurbText = $(".blurb-text");

    responseInner.html("");
    blurbText.hide();
    buy.hide();
    sell.hide();
    hold.hide();

    if (lineChart) {
        lineChart.destroy();
        lineChart2.destroy();
        lineChart3.destroy();
    }

    if (ticker === "select") {
        responseInner.append("<p>Please select a company...</p>")
        return false
    }

    $("#loading-overlay").show()

    let data = { "ticker": ticker };
    let endpoint = "http://2ca418b3-us-south.lb.appdomain.cloud:5000/inference"
    let ajax = $.ajax({
        type: "GET",
        url: endpoint,
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

function get_test_rsme(res){
    return res.test_rmse.toFixed(2);
}

function get_train_rsme(res){
    return parseFloat(res.train_rmse).toFixed(2);
}

function get_y_test(res){
    res_list = []
    res.y_test.forEach(el => {
      res_list.push(el[0])
    });
    return res_list;
}

function get_y_test_pred(res){
    return res.y_test_pred;
}

function get_dates(res){
    return res.dates;
}

function updateFrontend(ajax){
    ajax.done(function(res){

    test_rsme = get_test_rsme(res)
    train_rsme = get_train_rsme(res)
    actuals = get_y_test(res)
    predictions = get_y_test_pred(res)
    dates = get_dates(res)
    daysToPred = Math.abs(actuals.length - predictions.length)

    numDaysInPastLong = 200
    dates_long = dates.slice(dates.length - numDaysInPastLong, dates.length);
    actuals_long = actuals.slice(actuals.length - numDaysInPastLong, actuals.length);
    predictions_long = predictions.slice(predictions.length - numDaysInPastLong - daysToPred, predictions.length - daysToPred);

    numDaysInPastShort = 5
    dates_short = dates.slice(dates.length - numDaysInPastShort, dates.length);
    actuals_short = actuals.slice(actuals.length - numDaysInPastShort, actuals.length);
    predictions_short = predictions.slice(predictions.length - numDaysInPastShort - daysToPred, predictions.length - daysToPred);

    numDaysInPastImmedidate = 8
    dates_immed = dates.slice(dates.length - numDaysInPastImmedidate + daysToPred, dates.length);
    actuals_immed = actuals.slice(actuals.length - numDaysInPastImmedidate + daysToPred, actuals.length);
    predictions_immed_future = predictions.slice(predictions.length - numDaysInPastImmedidate, predictions.length);

    actual_last = actuals_immed[actuals_immed.length-1]
    prediction_last =  predictions_immed_future[predictions_immed_future.length-1]

    // Get future dates
    current = dates_immed[dates_immed.length - 1]
    dates_immed[dates_immed.length - 1] = "YESTERDAY"
    currentAsDate = new Date(current)
    for (let i = 1; i < daysToPred + 1; i++) {
        var newDate = new Date();
        newDate.setDate(currentAsDate.getDate() + i);
        newDateFormatted = newDate.getMonth()+1 +"/"+ newDate.getDate() +"/"+ newDate.getFullYear()
        console.log(newDateFormatted)
        dates_immed.push(newDateFormatted)
    }

    // Show prediction recommendation
    if (prediction_last > actual_last) {
        buy.show();
    } else if (prediction_last < actual_last) {
        sell.show();
    } else {
        hold.show();
    }

    $("#rmse").html("<p>Train RMSE: "+train_rsme+" vs Test RMSE: "+test_rsme+"</p>")

    let rsme_abs = Math.abs(test_rsme - train_rsme).toFixed(2)
    conf_str = "very"
    if (rsme_abs > 5) {
        conf_str = "not " + conf_str
    }

    $("#confidence").html("<p>With an absolute difference of the two RSMEs being "+rsme_abs+" we're "+ conf_str +" confident in this prediction! </p>")

    blurbText.show()

    let actual_colour = "#1767b4"
    let pred_colour = "#d60847"

    let ctx = $("#lineChart")[0].getContext('2d');
    let ctx_line_width = 1.5
    lineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates_long,
            datasets: [
                {
                    borderWidth: ctx_line_width,
                    label: "Actual",
                    data: actuals_long,
                    fill: false,
                    borderColor: actual_colour,
                    lineTension: 0.1
                },
                {
                    borderWidth: ctx_line_width,
                    label: "Predicted",
                    data: predictions_long,
                    fill: false,
                    borderColor: pred_colour,
                    lineTension: 0.1
                },
            ]
        },
        options: {
            elements: {
                point:{
                    radius: 0
                }
            },
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
                        text: "Past "+numDaysInPastLong+" Days Predictions",
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
            labels: dates_short,
            datasets: [
                {
                    label: "Actual",
                    data: actuals_short,
                    fill: false,
                    borderColor: actual_colour,
                    lineTension: 0.1
                },
                {
                    label: "Predicted",
                    data: predictions_short,
                    fill: false,
                    borderColor: pred_colour,
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
                        text: "Past "+numDaysInPastShort+" Days",
                        padding: 10,
                        font: {
                            size: 20
                        }
                    }
                }
            }
        }
    });

    let ctx3 = $("#lineChart3")[0].getContext('2d');
        lineChart3 = new Chart(ctx3, {
            type: "line",
            data: {
                labels: dates_immed,
                datasets: [
                    {
                        label: "Actual",
                        data: actuals_immed,
                        fill: false,
                        borderColor: actual_colour,
                        lineTension: 0.1
                    },
                    {
                        label: "Predicted",
                        data: predictions_immed_future,
                        fill: false,
                        borderColor: pred_colour,
                        lineTension: 0.1
                    }
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