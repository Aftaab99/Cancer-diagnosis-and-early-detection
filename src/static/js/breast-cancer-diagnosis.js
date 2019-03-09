$(document).ready(function () {

    var onCompletion = (res) => {
        console.log('Done')
        res = JSON.parse(res);
        console.log(res.image)
        res = res.image
        $("#image-box").attr("src", "data:image/jpeg;charset=utf-8;base64," + res);
    }

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
            url: '/diagnosis/breast_cancer'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });

    $('#select-image-form').submit((event) => {
        event.preventDefault();
        console.log('We are here!')
        var pb = $('#progress-bar').show();
        console.log('Form:' + $("#breast-cancer-diagnosis-image")[0].files[0])
        var form_data = new FormData();
        console.log(form_data)
        form_data.append('use_random', 0);
        form_data.append('breast-cancer-diagnosis-image', $("#breast-cancer-diagnosis-image")[0].files[0])
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
            url: '/diagnosis/breast_cancer'
        }).done(function (res) {
            console.log('POST request sent')
            onCompletion(res);
        });
    });

});