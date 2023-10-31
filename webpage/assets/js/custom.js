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
    $('.abstract').addClass('d-none');
    $('.Introduction').addClass('d-none');
    $('.RelatedWork').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.refrences').addClass('d-none');
    $('.Results').removeClass('d-none');
  });

  $( ".abstractHead" ).click(function() {
    $('.Introduction').addClass('d-none');
    $('.RelatedWork').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.refrences').addClass('d-none');
    $('.abstract').removeClass('d-none');
  });

  $( ".contributionHead" ).click(function() {
    $('.abstract').addClass('d-none');
    $('.RelatedWork').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.refrences').addClass('d-none');
    $('.Introduction').removeClass('d-none');
  });

  $( ".probStatHead" ).click(function() {
    $('.abstract').addClass('d-none');
    $('.Introduction').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.refrences').addClass('d-none');
    $('.RelatedWork').removeClass('d-none');
  });

  $( ".methodHead" ).click(function() {
    $('.abstract').addClass('d-none');
    $('.Introduction').addClass('d-none');
    $('.RelatedWork').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.refrences').addClass('d-none');
    $('.Methodology').removeClass('d-none');
  });

  $( ".refrencesHead" ).click(function() {
    $('.abstract').addClass('d-none');
    $('.Introduction').addClass('d-none');
    $('.RelatedWork').addClass('d-none');
    $('.Results').addClass('d-none');
    $('.Methodology').addClass('d-none');
    $('.refrences').removeClass('d-none');
  });

  $( ".navMenuIconWrapper a" ).click(function() {
    $('.navList').toggleClass('d-block');
  });
