import React from 'react'
import "./Spinner.css"
const Spinner = () => {
  return (
    <div className="container">
      <div className='spinner'></div>
      <p className="load-text">Loading....</p>
    </div>
  )
}

export default Spinner
