const messagebar =document.querySelector(".bar-wrapper-inp");
const sendbtn=document.querySelector(".bar-wrapper-btn");
const messagebox=document.querySelector(".message-box");


messagebar.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      if(messagebar.value.length>0){
        let message=`<div class="chat-message">
        <img src="image.png">
        <span>
            ${messagebar.value}
        </span>
        
    </div>`
    messagebar.value="";
    messagebox.insertAdjacentHTML("beforeend",message);
      document.getElementById("myBtn").click();
    }
  }});


sendbtn.onclick=function(){
      if(messagebar.value.length>0){
        let message=`<div class="chat-message">
        <img src="image.png">
        <span>
            ${messagebar.value}
        </span>
        
    </div>`
    messagebar.value="";
    messagebox.insertAdjacentHTML("beforeend",message);
      }
}