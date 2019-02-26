$(document).ready(function () {
    var onCompletion = function (response) {
        console.log(response)
        if(response.error)
            return;
        console.log(response.binary_prediction)
        let binary_pred = response.binary_prediction;
        let b_prob = response.binary_probability;
        let multiclass_pred = response.multiclass_prediction;
        let m_prob = response.multiclass_probablity;
        if (binary_pred == 1) {
            let result_msg = `<p>The drug corresponding to the fingerprint 
                can be used as a protein inhibitor(confidence:${b_prob}%). 
                The drug belongs to the class <b>${multiclass_pred}</b>(confidence:${m_prob}).</p>`;
            $('#result').removeClass('text-danger').addClass('text-success').html(result_msg);
        }
        else {
            let result_msg = '<p>The drug corresponding to the fingerprint cannot be used as a protein inhibitor';
            $('#result').removeClass('text-success').addClass('text-danger').html(result_msg);
        }
        $('#result-heading').show();
        $('#result-row').show();
    }
    $('#progress-bar').hide();
    $('#result-heading').hide();
    $('#result-row').hide();
    $('#use-random-btn').click(function (event) {
        event.preventDefault();
        $.ajax({
            type: 'POST',
            dataType: 'json',
            data: {
                use_random: 1
            },
            url: '/drug_discovery/protein_inhibitors'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });

    $('#dd-form').submit(function (event) {
        event.preventDefault();
        var pb = $('#progress-bar').show();

        var form_data = new FormData($('form')[0]);
        console.log(form_data)
        $.ajax({
            xhr: function () {
                var xhr = new XMLHttpRequest();
                xhr.upload.addEventListener('progress', function (e) {
                    console.log('Length computed')

                    if (e.lengthComputable) {
                        console.log('Length computed')
                        var pb = $('#progress-bar-inner');
                        let perc = Math.round((e.loaded / e.total)*100);
                        console.log(perc)
                        pb.css('width', perc+'%').html(`${perc}%`).attr('aria-valuenow', perc);
                    }
                });
                return xhr;

            },
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            url: '/drug_discovery/protein_inhibitors'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });
});