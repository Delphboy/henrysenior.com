let isHeaderDisplayed = true;

/**
 * EVENT HANDLER: Handle Keypress for the previousSearchQuery bar
 * If the key is RETURN: short circuit the form return and run AJAX previousSearchQuery
 * else: Run live previousSearchQuery
 */
function handleKeyPress(e, val) {
    if(e.keyCode === 27)
    {
        isHeaderDisplayed = !isHeaderDisplayed;
    }

    if(isHeaderDisplayed)
        updateHeader();
    else
        hideHeader();
}

function updateHeader()
{
    let header = document.getElementById('toolbar');
    let input = document.getElementById('textfield').value;

    header.innerHTML = 
        "<p>Words: " + wordCount(input) + " Characters: " + input.length + "</p>";
}

function hideHeader()
{
    let header = document.getElementById('toolbar');
    header.innerHTML = "";
}

function wordCount(str)
{ 
    return str.split(' ')
    .filter(function(n) { return n != '' })
    .length;
}