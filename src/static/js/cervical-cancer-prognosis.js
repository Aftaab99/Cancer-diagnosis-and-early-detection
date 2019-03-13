$(document).ready(function () {

    $('#result-heading').hide();
    $('#result-row').hide();
    
    var onCompletion = (res) => {
        $('#result-heading').show();
        $('#result-row').show();
        let result_p = (test_name)=>`Based on the modelled ${test_name} test, you are on the verge of developing cervical cancer.`
        let result_n = (test_name)=>`Based on the modelled ${test_name} test, you are unlikley to develop cervical cancer`
        let sch = $('#sch')
        let bio = $('#bio')
        let cit = $('#cit')
        let hins = $('#hins')

        if (res.sch == 1) {
            sch.html(result_p('Schiller'));
            sch.removeClass('text-success').addClass('text-danger')
        }
        else {
            sch.html(result_n('Schiller'));
            sch.removeClass('text-danger').addClass('text-success')
        }

        if (res.bio == 1) {
            bio.html(result_p('Biopsy'))
            bio.removeClass('text-success').addClass('text-danger')
        }
        else {
            bio.html(result_n('Biopsy'));
            bio.removeClass('text-danger').addClass('text-success')
        }

        if (res.cit == 1) {
            cit.html(result_p('Citology'))
            cit.removeClass('text-success').addClass('text-danger')
        }
        else {
            cit.html(result_n('Citology'));
            cit.removeClass('text-danger').addClass('text-success')
        }

        test_name = 'Hinselmann'
        if (res.hins == 1) {
            hins.html(result_p('Hinselmann'))
            hins.removeClass('text-success').addClass('text-danger')
        }
        else {
            hins.html(result_n('Hinselmann'));
            hins.removeClass('text-danger').addClass('text-success')
        }
    }

    $('form[name="cervical-cancer-prognosis-form"]').submit(function (event) {
        event.preventDefault();
        var form = $('form[name="cervical-cancer-prognosis-form"]')[0];
        if (form.checkValidity() === false) {
            console.log('Not valid!!')
            return;
        }
        else {
            let form_data = new FormData();

            form_data.append('age', $('input[name="age"]').val());
            form_data.append('no_sexual_partners', $('input[name="no_sexual_partners"]').val());
            form_data.append('first_sex_age', $('input[name="first_sex_age"]').val());
            form_data.append('no_of_pregnancies', $('input[name="no_of_pregnancies"]').val());
            form_data.append('no_of_std', $('input[name="no_of_std"]').val());
            form_data.append('diagnosed_with_cin', $('input[name="diagnosed_with_cin"]').val());
            form_data.append('diagnosed_with_hpv', $('input[name="diagnosed_with_hpv"]').val());

            $.ajax({
                type: 'POST',
                data: form_data,
                processData: false,
                contentType: false,
                url: '/prognosis/cervical_cancer'
            }).done(function (response) {
                onCompletion(response);
            });
        }

    });



});