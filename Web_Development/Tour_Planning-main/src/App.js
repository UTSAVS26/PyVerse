import { useState } from 'react';
import Cards from "./components/Cards"
import data from './data'


function App() {
  const [tours,setTours]=useState(data)
  function removeHandler(id){
    setTours(tours.filter(tour=>
       tour.id !== id
    ))
  }

  if(tours.length===0){
    return (
      <>
        <div className='refresh-container'>
          <h1 className=''>No Tour Left</h1>
          <button onClick={()=>{
            setTours(data);
          }} className='refresh-btn'>Refresh</button>
        </div>
      </>
    )
  }
  return (
    <>
      <div className="container">
        <h1 className='heading'>Plan With Shariq</h1>
        <Cards tours={tours} removeHandler={removeHandler}/>
      </div>
    </>
  );
}

export default App;
