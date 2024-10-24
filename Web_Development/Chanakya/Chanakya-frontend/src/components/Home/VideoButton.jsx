import React from "react";

export default function VideoButton() {
  return (
    <button className="d-flex align-items-center justify-content-center text-white" style={{
      width: '80px',
      height: '80px',
      borderRadius: '50%',
      background: 'linear-gradient(30deg, rgb(255, 130, 0) 20%, rgb(255, 38, 0) 80%)',
      transition: 'all 0.3s ease-in-out 0s',
      boxShadow: 'rgba(70, 68, 68, 0.698) 0px 0px 0px 0px',
      animation: 'pulse 1.2s cubic-bezier(0.8, 0, 0, 1) infinite',
      border: '0',
    }}>
      <svg
        viewBox="0 0 448 512"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        width="26px"
        style={{ color: 'currentColor' }}
      >
        <path
          d="M424.4 214.7L72.4 6.6C43.8-10.3 0 6.1 0 47.9V464c0 37.5 40.7 60.1 72.4 41.3l352-208c31.4-18.5 31.5-64.1 0-82.6z"
          fill="currentColor"
        ></path>
      </svg>
    </button>
  );
}
