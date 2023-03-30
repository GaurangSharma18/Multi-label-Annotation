$(document).ready(function() {

  // Sidebar toggle behavior
  //$('#sidebarCollapse').on('click', function() {
  //  $('#sidebar, #content').toggleClass('active');

    //$('.contentWrapper').toggleClass('w-100');
    //$('.frontPageImages').toggleClass('d-none');
   
   
   // pageHeight = $(window).height();
    //console.log(pageHeight)
    //$('.paperBody').css('height',pageHeight-pageHeight*0.1)
      

  });


  $( ".resultHead" ).click(function() {
    console.log('clicked')
    $('.abstract').addClass('d-none');
    $('.Contributions').addClass('d-none');
    $('.ProblemStatemen').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').removeClass('d-none');
  });

  $( ".abstractHead" ).click(function() {
    console.log('clicked')
    $('.Contributions').addClass('d-none');
    $('.ProblemStatemen').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.abstract').removeClass('d-none');
  });

  $( ".contributionHead" ).click(function() {
    console.log('clicked')
    $('.abstract').addClass('d-none');
    $('.ProblemStatemen').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.Contributions').removeClass('d-none');
  });

  $( ".probStatHead" ).click(function() {
    console.log('clicked')
    $('.abstract').addClass('d-none');
    $('.Contributions').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.ProblemStatemen').removeClass('d-none');
  });

  $( ".methodHead" ).click(function() {
    console.log('clicked')
    $('.abstract').addClass('d-none');
    $('.Contributions').addClass('d-none');
    $('.ProblemStatemen').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.Methodology').removeClass('d-none');
  });