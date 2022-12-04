$("#ticker-selector-submit").click(function () {

    let startDate = new Date();
    let responseHolder = $("#response")
    let ticker = $("#ticker").val()

    responseHolder.html("");

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

    var endDate = new Date();
    var seconds = (endDate - startDate) / 1000
    if (seconds < 1) {
        var delay = 1000;
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
        $("#loading-overlay").hide();
    });

    ajax.fail(function(res){
        $("#response").html(res.responseJSON.message)
         $("#loading-overlay").hide();
    });
}