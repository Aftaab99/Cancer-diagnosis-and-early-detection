$(document).ready(function () {

    var onCompletion = (res) => {
        console.log('Done')
        res = JSON.parse(res);
        prediction = res.prediction;
        console.log("Prediction="+prediction+",type="+(typeof prediction)+"equals="+(prediction==1))
        res1 = res.image;
        $("#image-box").attr("src", "data:image/jpeg;charset=utf-8;base64," + res1);
        if (prediction == 1) {
            let result_msg = `<p>The image was diagnosed to be melanoma positive</p>`;
            $('#result').removeClass('text-success').addClass('text-danger').html(result_msg);
        }
        else {
            let result_msg = '<p>The image was diagnosed to be melanoma negative</p>';
            $('#result').removeClass('text-danger').addClass('text-success').html(result_msg);
        }
        $('#result-heading').show();
        $('#result-row').show();
    }

    $('#result-heading').hide();
    $('#result-row').hide();
    $('#progress-bar').hide();
    
    $('#random-form').submit(function (event) {
        event.preventDefault();
        console.log('event started')
        $.ajax({
            type: 'POST',
            data: {
                use_random: 1
            },
            mimeType: "text/plain; charset=x-user-defined",
            url: '/diagnosis/skin_cancer'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });

    $('#select-image-form').submit((event) => {
        event.preventDefault();
        console.log('We are here!')
        var pb = $('#progress-bar').show();
        var form_data = new FormData();
        console.log(form_data)
        var file = $("#skin-cancer-diagnosis-image")[0].files[0];
        var age = $('#age-select').val();
        var gender = $('#gender-select').val();

        if ((isNaN(age)) || (age <= 0)) {
            $('#age-select').addClass('is-invalid');
            return;
        }
        if (file === undefined) {
            $('#skin-cancer-diagnosis-image').addClass('is-invalid'); 
            return;
        }

        form_data.append('use_random', 0);
        form_data.append('skin-cancer-diagnosis-image', $("#skin-cancer-diagnosis-image")[0].files[0])

        form_data.append('age', age);
        form_data.append('gender', gender);
        $.ajax({
            xhr: function () {
                var xhr = new XMLHttpRequest();
                xhr.upload.addEventListener('progress', function (e) {
                    console.log('Length computed')

                    if (e.lengthComputable) {
                        console.log('Length computed')
                        var pb = $('#progress-bar-inner');
                        let perc = Math.round((e.loaded / e.total) * 100);
                        console.log(perc)
                        pb.css('width', perc + '%').html(`${perc}%`).attr('aria-valuenow', perc);
                    }
                });
                return xhr;
            },
            type: 'POST',
            data: form_data,
            contentType: false,
            processData: false,
            mimeType: "text/plain; charset=x-user-defined",
            url: '/diagnosis/skin_cancer'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });

});