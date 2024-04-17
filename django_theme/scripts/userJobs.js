import $ from 'jquery';
console.log(process.cwd())
// fetch("user_jobs.json")
//     .then((response) => response.json())
//     .then((json) => console.log(json));
const s = require('./user_jobs.json')
console.log(s[0])
s.forEach(e => {
    console.log(e[1])
});
var previous = null;
    var current = null;
    setInterval(function() {
        $.getJSON("user_jobs.json", function(json) {
            current = JSON.stringify(json);            
            if (previous && current && previous !== current) {
                console.log('refresh');
                location.reload();
            }
            previous = current;
        });                       
    }, 2000);   