import React from 'react'
import LoginForm from '../../components/LoginForm/LoginForm'
import './LoginPage.css'

const LoginPage = ({loginHandler}) => {
  return (
    <>
      <LoginForm loginHandler={loginHandler}/>
    </>
  )
}

export default LoginPage
